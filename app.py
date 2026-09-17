import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import gradio as gr
import httpx
import pandas as pd
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET_URL = os.environ.get(
    "ICMR_HF_DATASET_URL",
    "https://huggingface.co/datasets/rehuuuu/icrm-hitek-fulldb/resolve/main",
).rstrip("/")

PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
TIMEOUT = int(os.environ.get("ICMR_TIMEOUT", "60"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

PARQUET_FILES = {
    "phone": [f"{HF_DATASET_URL}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_DATASET_URL}/idx_aadhar.{i}.parquet" for i in range(7)],
}

_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="search")


# ── Parquet Download ────────────────────────────────────────────────────────
async def download_parquet(url: str) -> pd.DataFrame:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            buffer = BytesIO(r.content)
            table = pq.read_table(buffer)
            return table.to_pandas()
    except Exception as e:
        print(f"❌ {url}: {e}")
        return pd.DataFrame()


async def search_in_parquet(field: str, value: str, limit: int = 10) -> list:
    results = []
    if field in ("phoneNumber", "otherNumber"):
        files = PARQUET_FILES["phone"]
    elif field == "aadharNumber":
        files = PARQUET_FILES["aadhar"]
    else:
        files = PARQUET_FILES["phone"]

    for url in files[:3]:
        try:
            df = await download_parquet(url)
            if df.empty:
                continue
            if field in df.columns:
                mask = df[field].astype(str).str.strip() == value
                matches = df[mask]
                for _, row in matches.iterrows():
                    results.append(row.to_dict())
                    if len(results) >= limit:
                        return results
        except Exception as e:
            print(f"⚠️ {url}: {e}")
            continue
    return results


# ── Dedup ───────────────────────────────────────────────────────────────────
def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None or pd.isna(raw):
            continue
        v = str(raw).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        connected.append({"field": field, "value": v})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen, out = {}, []
    for r in rows:
        ph = str(r.get("phoneNumber", "")).strip()
        ad = str(r.get("aadharNumber", "")).strip()
        key = (ph, ad) if ph or ad else (
            str(r.get("name", "")), str(r.get("fathersName", ""))
        )
        n = seen.get(key, 0)
        if n < DUPLICATE_CAP:
            seen[key] = n + 1
            rec = dict(r)
            rec["connected_numbers"] = _connected_numbers(rec)
            out.append(rec)
    return out


# ── Search ──────────────────────────────────────────────────────────────────
def _unified_search_sync(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q or not q.isdigit() or len(q) < 8:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []

    if len(q) in (10, 11):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rows = loop.run_until_complete(search_in_parquet("phoneNumber", q, limit))
            loop.close()
            if rows:
                all_rows.extend(rows)
                searched.append("phoneNumber")
        except Exception as e:
            print(f"Phone error: {e}")

    if len(q) == 12:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rows = loop.run_until_complete(search_in_parquet("aadharNumber", q, limit))
            loop.close()
            if rows:
                all_rows.extend(rows)
                searched.append("aadharNumber")
        except Exception as e:
            print(f"Aadhar error: {e}")

    all_rows = _cap_duplicates(all_rows)[:limit]
    return {
        "query": q, "searched_fields": searched,
        "count": len(all_rows), "results": all_rows,
    }


def _run_field_search_sync(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        return {"field": field, "value": value, "mode": mode,
                "count": 0, "results": [], "error": "Unknown field"}
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        rows = loop.run_until_complete(search_in_parquet(field, value, limit))
        loop.close()
        rows = _cap_duplicates(rows)[:limit]
        return {"field": field, "value": value, "mode": mode,
                "count": len(rows), "results": rows}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode,
                "count": 0, "results": [], "error": str(e)}


# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                r = await client.get(url)
                print(f"[Pinger] {r.status_code}")
            except Exception as e:
                print(f"[Pinger] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 ICMR Search API started!")
    asyncio.create_task(pinger())
    yield
    print("👋 Shutting down...")


# ── FastAPI ─────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR Search API", lifespan=lifespan)


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "dataset": "rehuuuu/icrm-hitek-fulldb",
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | @api_wallah",
    }


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": "rehuuuu/icrm-hitek-fulldb",
        "searchable_fields": ["phoneNumber", "aadharNumber"],
    }


@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    aadhar: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    if aadhar:
        q_val, field = aadhar.strip(), "aadharNumber"
    elif mobile:
        q_val, field = mobile.strip(), "phoneNumber"
    elif q:
        q_val = q.strip()
    else:
        raise HTTPException(422, "Provide q, mobile, or aadhar")

    if not q_val:
        raise HTTPException(422, "Query cannot be empty")

    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(
            pool, _run_field_search_sync, field, q_val, mode, limit
        )
    else:
        data = await loop.run_in_executor(
            pool, _unified_search_sync, q_val, limit
        )

    result = {"success": bool(data.get("count", 0) > 0), **data, "number": q_val}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False, default=str)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _run_field_search_sync,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return {"searches": len(req.queries), "results": list(results)}


# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for f in SEARCH_FIELDS:
        v = row.get(f, "")
        if v and str(v) != "nan":
            lines.append(f"**{f}:** {v}")
    cn = row.get("connected_numbers", [])
    if cn:
        lines.append("**connected:** " + ", ".join(f"{c['field']}={c['value']}" for c in cn))
    return "\n\n".join(lines)


def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh daalo — phone ya aadhar."
    q = query.strip()
    data = _unified_search_sync(q, int(limit))
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))

    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found**"

    parts = [f"🔍 **Query:** `{q}` | **Found:** {count} | **Searched:** {searched}"]
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")
    return "\n\n---\n\n".join(parts)


def build_ui():
    with gr.Blocks(title="ICMR Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API")
        gr.Markdown("Search **rehuuuu/icrm-hitek-fulldb**")
        with gr.Row():
            qi = gr.Textbox(label="Search Query", placeholder="Phone ya Aadhaar", lines=1)
            ls = gr.Slider(1, 20, value=5, step=1, label="Max Results")
        btn = gr.Button("🔍 Search", variant="primary", size="lg")
        out = gr.Markdown(label="Results")
        btn.click(fn=search_ui, inputs=[qi, ls], outputs=out)
        qi.submit(fn=search_ui, inputs=[qi, ls], outputs=out)
        gr.Markdown("---\n<div style='text-align:center;color:#888;'>👨‍💻 @kzr0x | 📢 @api_wallah</div>")
    return demo


demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
