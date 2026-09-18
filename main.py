import os
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ============================================================
#  CONFIG (kuch change karne ki zaroorat nahi)
# ============================================================
HF_NAMESPACE = "bronx-ultra"
HF_BUCKET    = "icrm-hitek-full-db-mixed-bucket"

PARALLELISM      = 2
THREADS_PER_CONN = 2
DUPLICATE_CAP    = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

# ============================================================
#  CONNECTION POOL
# ============================================================
_conns = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    # STEP 1: temp directory (Render ke liye zaroori)
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")

    # STEP 2: extensions load karo
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # ========================================================
    #  STEP 3: ⭐ S3 SECRET YAHAN LAGTA HAI ⭐
    # ========================================================
    con.execute("""
        CREATE OR REPLACE SECRET hf_s3 (
            TYPE s3,
            KEY_ID 'dummy',
            SECRET 'dummy',
            ENDPOINT 's3.hf.co/bronx-ultra',
            URL_STYLE 'path',
            REGION 'us-east-1'
        )
    """)

    # ========================================================
    #  STEP 4: ⭐ S3 PATHS YAHAN LAGTE HAIN ⭐
    # ========================================================
    phone_urls = [
        f"s3://icrm-hitek-full-db-mixed-bucket/idx_phone.{i}.parquet"
        for i in range(7)
    ]
    aadhar_urls = [
        f"s3://icrm-hitek-full-db-mixed-bucket/idx_aadhar.{i}.parquet"
        for i in range(7)
    ]

    # STEP 5: views banao
    phone_lst  = ", ".join(f"'{u}'" for u in phone_urls)
    aadhar_lst = ", ".join(f"'{u}'" for u in aadhar_urls)

    con.execute(f"CREATE OR REPLACE VIEW people_phone  AS SELECT * FROM read_parquet([{phone_lst}])")
    con.execute(f"CREATE OR REPLACE VIEW people_aadhar AS SELECT * FROM read_parquet([{aadhar_lst}])")

    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = getattr(_thread_local, "id", None)
    if ident is None:
        with _conns_lock:
            ident = len(_conns)
            _thread_local.id = ident
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ============================================================
#  DEDUP HELPERS
# ============================================================
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


# ============================================================
#  SEARCH
# ============================================================
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber":
            view = "people_phone"
        elif field == "aadharNumber":
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not (q.isdigit() and len(q) >= 8):
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []

    r = _run_field_search("phoneNumber", q, "exact", limit)
    all_rows.extend(r["results"])
    searched.append("phoneNumber")

    if not all_rows:
        r = _run_field_search("aadharNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("aadharNumber")

    all_rows = _cap_duplicates(all_rows)[:limit]
    return {"query": q, "searched_fields": searched, "count": len(all_rows), "results": all_rows}


# ============================================================
#  FASTAPI
# ============================================================
app = FastAPI(title="ICMR + HITEK Search API")


@app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "indexes": {"phone": True, "aadhar": True},
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
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
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)

    result = {"success": bool(data["count"]), **data, "number": q_val, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
