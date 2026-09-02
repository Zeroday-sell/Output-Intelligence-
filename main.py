import asyncio
import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Literal
import duckdb
import httpx
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import uvicorn

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

HF_INDEX_BASE = os.getenv(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Nasskeke/icrm-hitek-full-db-mixed/resolve/main"
).rstrip("/")

INDEX_SOURCE = os.getenv("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.getenv("ICMR_PARALLEL", "4"))
THREADS_PER_CONN = int(os.getenv("ICMR_THREADS_PER_CONN", "2"))
PORT = int(os.getenv("PORT", "7860"))
DUPLICATE_CAP = int(os.getenv("DUPLICATE_CAP", "2"))
MAX_LIMIT = 100
DEFAULT_LIMIT = 10
MAX_BATCH_QUERIES = 50

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ──────────────────────────────────────────────
# DUCKDB THREAD-LOCAL CONNECTIONS
# ──────────────────────────────────────────────

_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
executor = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET enable_object_cache=true;")
    
    # Create views for phone and Aadhaar indexes
    for kind, urls in REMOTE_INDEXES.items():
        view_name = f"people_{kind}"
        url_list = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet([{url_list}])")
    
    con.execute(f"SET threads={THREADS_PER_CONN};")
    return con

def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid

def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]

# ──────────────────────────────────────────────
# DEDUP & CONNECTED NUMBERS
# ──────────────────────────────────────────────

def _person_key(row: Dict[str, Any]) -> tuple:
    ph = str(row.get("phoneNumber") or "").strip()
    ad = str(row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (str(row.get("name") or "").strip(), str(row.get("fathersName") or "").strip())

def _connected_numbers(row: Dict[str, Any]) -> List[Dict[str, str]]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ──────────────────────────────────────────────
# SEARCH LOGIC
# ──────────────────────────────────────────────

def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    
    v = value.replace("'", "''")
    limit = min(limit, MAX_LIMIT)
    
    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
            sql = f"SELECT * FROM {view} WHERE phoneNumber = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
            sql = f"SELECT * FROM {view} WHERE aadharNumber = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
        elif field == "otherNumber":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        elif field in ("name", "fathersName", "address", "district", "pincode", "state", "town", "source"):
            # Search on phone index (has all data, sorted by phone but can scan)
            view = "people_phone"
            escaped = v.replace("%", r"\%").replace("_", r"\_")
            sql = f"SELECT * FROM {view} WHERE {field} ILIKE '%{escaped}%' LIMIT {limit * DUPLICATE_CAP + 20}"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
    
    elif mode == "contains":
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
            sql = f"SELECT * FROM {view} WHERE phoneNumber LIKE '%{v2}%' LIMIT {limit * DUPLICATE_CAP + 20}"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
            sql = f"SELECT * FROM {view} WHERE aadharNumber LIKE '%{v2}%' LIMIT {limit * DUPLICATE_CAP + 20}"
        elif field in ("name", "fathersName", "address", "district", "pincode", "state", "town", "source"):
            view = "people_phone"
            sql = f"SELECT * FROM {view} WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    con = _get_conn()
    try:
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": str(e)}

def _unified_search(q: str, limit: int = DEFAULT_LIMIT) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    
    if is_num:
        all_rows = []
        searched = []
        
        # Try phone first
        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        
        # Try Aadhaar if no phone results
        if not all_rows and _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")
        
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q,
            "searched_fields": searched,
            "count": len(all_rows),
            "results": all_rows
        }
    else:
        # Name search
        r = _run_field_search("name", q, "contains", limit)
        return {
            "query": q,
            "searched_fields": ["name"],
            "count": r["count"],
            "results": r["results"]
        }

# ──────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────

fastapi_app = FastAPI(
    title="ICMR + HITEK Search API",
    description="Search 5.01 billion Indian citizen records",
    version="2.0.0"
)

class BatchQueryItem(BaseModel):
    field: Optional[str] = Field(None, description="Field to search. Auto-detected if not provided.")
    value: str = Field(..., description="Value to search for")
    limit: Optional[int] = Field(DEFAULT_LIMIT, description="Max results to return")
    mode: Optional[str] = Field("exact", description="Search mode: exact or contains")

class BatchSearchRequest(BaseModel):
    queries: List[BatchQueryItem] = Field(..., min_length=1, max_length=MAX_BATCH_QUERIES)
    limit: Optional[int] = Field(DEFAULT_LIMIT)

@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "version": "2.0.0",
        "records": 5_009_587_740,
        "indexes": {
            "phone": _idx_ready("phone"),
            "aadhar": _idx_ready("aadhar")
        },
        "index_source": INDEX_SOURCE,
        "columns": SEARCH_FIELDS,
        "endpoints": {
            "/search": "Primary search endpoint",
            "/search/parallel": "Batch search (POST, up to 50 queries)",
            "/health": "Health check",
            "/docs": "Swagger UI"
        },
        "developer": "@monster13",
       
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "indexes": {
            "phone": _idx_ready("phone"),
            "aadhar": _idx_ready("aadhar")
        },
        "index_source": INDEX_SOURCE,
        "active_connections": len(_conns),
        "timestamp": time.time()
    }

@fastapi_app.get("/search")
async def search(
    q: Optional[str] = Query(None, description="Unified query — auto-detects type"),
    mobile: Optional[str] = Query(None, description="Mobile alias (same as q)"),
    field: Optional[str] = Query(None, description="Specific field to search"),
    mode: str = Query("exact", description="Search mode: exact or contains"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    pretty: bool = Query(True)
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")
    
    loop = asyncio.get_running_loop()
    
    if field:
        data = await loop.run_in_executor(executor, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(executor, _unified_search, q_val, limit)
    
    result = {
        "success": bool(data.get("count", 0)),
        **data,
        "number": q_val,
        "total": data.get("count", 0)
    }
    
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchSearchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > MAX_BATCH_QUERIES:
        raise HTTPException(400, f"max {MAX_BATCH_QUERIES} queries per batch")
    
    loop = asyncio.get_running_loop()
    tasks = []
    
    for item in req.queries:
        field = item.field or "phoneNumber"
        value = item.value
        mode = item.mode or "exact"
        limit = item.limit or req.limit or DEFAULT_LIMIT
        tasks.append(loop.run_in_executor(executor, _run_field_search, field, value, mode, limit))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({
                "query": req.queries[i].value,
                "field": req.queries[i].field or "auto",
                "count": 0,
                "error": str(result),
                "results": []
            })
        else:
            processed.append(result)
    
    return Response(
        content=json.dumps({
            "total_queries": len(req.queries),
            "successful": sum(1 for r in results if not isinstance(r, Exception)),
            "failed": sum(1 for r in results if isinstance(r, Exception)),
            "results": processed
        }, indent=2, ensure_ascii=False),
        media_type="application/json"
    )

# ──────────────────────────────────────────────
# PINGER
# ──────────────────────────────────────────────

async def pinger():
    """Ping /health every 2 minutes to keep the app warm."""
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    await asyncio.sleep(10)
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                print(f"[Pinger] Status: {resp.status_code}")
        except Exception as e:
            print(f"[Pinger] Error: {e}")
        await asyncio.sleep(120)

@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())

# ──────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────

def format_result(row: Dict[str, Any]) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val and val != "null":
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Enter a phone, Aadhaar, or name to search."
    
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))
    
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found.**"
    
    header = f"🔍 **Query:** `{q}` | **Found:** {count} | **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(
        title="ICMR + HITEK Search",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .footer { text-align: center; color: #888; margin-top: 20px; }
        """
    ) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API", elem_classes="main-title")
        gr.Markdown("Search **5.01 billion records** — phone, Aadhaar, name & more", elem_classes="subtitle")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone, Aadhaar, or name...",
                    lines=1
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results"
                )
        
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        
        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output
        )
        
        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints:**
- `GET /search?q=<number>` — Phone/Aadhaar search
- `GET /search?q=<name>&mode=contains` — Name search
- `GET /search?field=<field>&q=<value>` — Field-specific search
- `GET /health` — Health check
- `GET /docs` — Swagger UI

**Dataset:** [Nasskeke/icrm-hitek-full-db-mixed](https://huggingface.co/datasets/Nasskeke/icrm-hitek-full-db-mixed)
            """)
        
        gr.Markdown(
            "---\n"
            "<div class='footer'>"
            "👨‍💻 **Developer:** @monster13 | 🥀 **Instagram:**@anime.boy.30"
            "</div>",
            elem_classes="footer"
        )
    
    return demo

# ──────────────────────────────────────────────
# MOUNT GRADIO
# ──────────────────────────────────────────────

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
