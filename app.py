import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
import gradio as gr
import httpx
from huggingface_hub import HfFileSystem   # ✅ bucket access ke liye
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# 🔑 APNA HF TOKEN YAHAN DAALO (ya Render env var se set karo)
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_PASTE_YOUR_TOKEN_HERE")

# ✅ Bucket details
HF_BUCKET_ID = "bronx-ultra/icrm-hitek-full-db-mixed-bucket"
HF_BUCKET_BASE = f"https://huggingface.co/buckets/{HF_BUCKET_ID}/resolve"
HF_BUCKET_DUCKDB = f"hf://buckets/{HF_BUCKET_ID}"   # HfFileSystem ke saath kaam karega

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

IDX_PHONE = "idx_phone"
IDX_AADHAR = "idx_aadhar"

# ✅ SAARI 7 files use ho rahi hain — 100% index coverage
REMOTE_INDEXES = {
    "phone": [f"{HF_BUCKET_DUCKDB}/{IDX_PHONE}.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_BUCKET_DUCKDB}/{IDX_AADHAR}.{i}.parquet" for i in range(7)],
}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    """Naya DuckDB connection banao — HfFileSystem register karke."""
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # ✅ HfFileSystem register karo — BUCKET access ke liye ZAROORI
    #    Token ke saath, taaki private/gated bucket bhi kaam kare
    fs = HfFileSystem(token=HF_TOKEN if HF_TOKEN.startswith("hf_") else None)
    con.register_filesystem(fs)

    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet([{lst}], union_by_name=true)"
        )
    con.execute(f"SET threads = {THREADS_PER_CONN}")
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


# ── Dedup & Connected Records ───────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _connected_numbers(row: dict) -> list[dict]:
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


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
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


# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode,
                    "count": 0, "results": []}
        sql = (
            f"SELECT * FROM {view} WHERE {field} = '{v}' "
            f"LIMIT {limit * DUPLICATE_CAP + 20}"
        )
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode,
                    "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (
            f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' "
            f"ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode,
            "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8

    if not is_num:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []

    # ✅ Pehle PHONE index (saari 7 files) — fast
    if _idx_ready("phone"):
        r = _run_field_search("phoneNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("phoneNumber")

    # ✅ Agar phone mein nahi mila toh AADHAR index (saari 7 files)
    if not all_rows and _idx_ready("aadhar"):
        r = _run_field_search("aadharNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("aadharNumber")

    # Enrich — connected numbers se extra records
    if all_rows:
        connected_searches = set()
        for row in all_rows[:3]:
            for nf in NUMBER_FIELDS:
                nv = row.get(nf)
                if nv and str(nv) not in connected_searches:
                    connected_searches.add(str(nv))
                    extra = _run_field_search(nf, str(nv), "exact", 2)
                    if extra["count"]:
                        all_rows.extend(extra["results"])
                        if nf not in searched:
                            searched.append(nf)

    all_rows = _cap_duplicates(all_rows)[:limit]

    return {
        "query": q, "searched_fields": searched,
        "count": len(all_rows), "results": all_rows,
    }


# ── FastAPI ────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "bucket": HF_BUCKET_ID,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@BRONX_ULTRA | credit @BRONX_ULTRA",
    }


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "bucket": HF_BUCKET_ID,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
    }


@fastapi_app.get("/debug")
def debug():
    """✅ PEHLE YE CHALAO — bucket aur token check karne ke liye."""
    try:
        con = _get_conn()
        # Phone index ki pehli file se sample
        sample = con.execute(
            f"SELECT * FROM read_parquet('{REMOTE_INDEXES['phone'][0]}', union_by_name=true) LIMIT 1"
        ).fetchdf()
        # Saari phone files count karo
        files = con.execute(
            f"SELECT COUNT(*) FROM glob('{HF_BUCKET_DUCKDB}/{IDX_PHONE}.*.parquet')"
        ).fetchone()[0]
        con.close()
        return {
            "status": "connected",
            "bucket": HF_BUCKET_ID,
            "duckdb_path": HF_BUCKET_DUCKDB,
            "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
            "phone_files_found": files,
            "columns": list(sample.columns) if not sample.empty else [],
        }
    except Exception as e:
        return {
            "status": "error",
            "bucket": HF_BUCKET_ID,
            "duckdb_path": HF_BUCKET_DUCKDB,
            "token_loaded": bool(HF_TOKEN and HF_TOKEN.startswith("hf_")),
            "error": str(e),
        }


@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")

    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(
            pool, _run_field_search, field, q_val, mode, limit
        )
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)

    result = {"success": bool(data["count"]), **data, "number": q_val,
              "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None,
                         ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _run_field_search,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps(
            {"searches": len(req.queries), "results": list(results)},
            indent=2, ensure_ascii=False, default=str,
        ),
        media_type="application/json",
    )


# ── Pinger (keeps app alive) ──────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] Status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")


@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())


# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val and str(val) != "nan":
            lines.append(f"**{field}:** {val}")

    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")

    return "\n\n".join(lines)


def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone, aadhar, ya name daalo."

    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))

    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found**."

    header = (
        f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  "
        f"**Searched:** {searched}\n\n---\n\n"
    )
    parts = []
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")

    return header + "\n\n---\n\n".join(parts)


def build_ui():
    with gr.Blocks(
        title="ICMR Search API",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .footer { text-align: center; color: #888; margin-top: 20px; }
        """
    ) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API", elem_classes="main-title")
        gr.Markdown(
            "Search **2.5 billion records** — phone, Aadhaar, name, address & more",
            elem_classes="subtitle",
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number, Aadhaar, ya name daalo...",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results",
                )

        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")

        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )

        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints** (via FastAPI):
- `GET /search?q=<number>` — Phone/Aadhaar search
- `GET /search?mobile=<number>` — Phone search (alias)
- `GET /health` — Health check
- `GET /debug` — Bucket connection test
- `GET /docs` — Swagger UI
            """)

        gr.Markdown(
            "---\n"
            "<div class='footer'>"
            "👨‍💻 **Developer:** @kzr0x  |  📢 **Channel:** @api_wallah"
            "</div>",
            elem_classes="footer",
        )

    return demo


# ── Mount Gradio on FastAPI ─────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"\n🚀 ICMR Search API chal raha hai: http://0.0.0.0:{port}")
    print(f"📖 Docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
