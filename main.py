from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import duckdb
import os
import threading
import time
import logging

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hitek")

# ---------- App ----------
app = FastAPI(title="Hitek Data Gateway", version="2.0")

# ---------- DuckDB ----------
con = duckdb.connect(database=":memory:")
try:
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET enable_object_cache=true;")
    log.info("[DuckDB] httpfs loaded successfully")
except Exception as e:
    log.error(f"[DuckDB] httpfs setup failed: {e}")

# ---------- Config ----------
HF_BASE = "https://huggingface.co/datasets/CutehackX/hitek-data-bucket/resolve/main"
SHARDS = list("0123456789")
loaded_shards = set()
load_lock = threading.Lock()

# ---------- Landing Page HTML ----------
LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hitek Data Gateway - LIVE</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #050505; color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
        #canvas-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }
        .overlay { 
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
            text-align: center; background: rgba(10, 10, 10, 0.85); padding: 50px; 
            border: 1px solid #00ffcc; border-radius: 12px; box-shadow: 0 0 30px rgba(0, 255, 204, 0.3); 
            backdrop-filter: blur(5px);
        }
        h1 { margin: 0 0 15px 0; font-size: 3.5em; text-transform: uppercase; letter-spacing: 6px; text-shadow: 0 0 15px #00ffcc; }
        p { font-size: 1.2em; margin: 8px 0; color: #ccc; }
        .highlight { color: #00ffcc; font-weight: bold; }
        .status-box { 
            margin-top: 30px; font-weight: bold; padding: 15px; 
            border-radius: 8px; background: rgba(0, 255, 204, 0.1); 
            border: 1px solid rgba(0, 255, 204, 0.5);
            font-size: 1.1em;
        }
        .blinking { animation: blinker 1.5s linear infinite; display: inline-block; }
        @keyframes blinker { 50% { opacity: 0; } }
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    <div class="overlay">
        <h1>SYSTEM ONLINE</h1>
        <p>API Gateway is <span class="highlight">Active & Secured</span></p>
        <p>Parquet Cloud Engine: <span class="highlight">Connected</span></p>
        <div class="status-box">
            <span class="blinking" style="color: #00ffcc;">●</span> HTTP 200 OK - LISTENING FOR QUERIES
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('canvas-container').appendChild(renderer.domElement);
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        for (let i = 0; i < 8000; i++) {
            vertices.push(THREE.MathUtils.randFloatSpread(3000));
            vertices.push(THREE.MathUtils.randFloatSpread(3000));
            vertices.push(THREE.MathUtils.randFloatSpread(3000));
        }
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        const material = new THREE.PointsMaterial({ color: 0x00ffcc, size: 2.5, transparent: true, opacity: 0.8 });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        camera.position.z = 1200;
        function animate() {
            requestAnimationFrame(animate);
            points.rotation.x += 0.0005;
            points.rotation.y += 0.001;
            renderer.render(scene, camera);
        }
        animate();
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""

# ---------- Background Shard Loader ----------
def load_shard(digit: str):
    """Load a single shard pair (main + alt) into DuckDB memory."""
    try:
        main_url = f"{HF_BASE}/final_master_shard_{digit}.parquet"
        con.execute(f"""
            CREATE OR REPLACE TABLE main_shard_{digit} AS
            SELECT * FROM read_parquet('{main_url}')
        """)
        log.info(f"[Shard {digit}] Main loaded")
    except Exception as e:
        log.error(f"[Shard {digit}] Main failed: {e}")

    try:
        alt_url = f"{HF_BASE}/alt_master_shard_{digit}.parquet"
        con.execute(f"""
            CREATE OR REPLACE TABLE alt_shard_{digit} AS
            SELECT * FROM read_parquet('{alt_url}')
        """)
        log.info(f"[Shard {digit}] Alt loaded")
    except Exception as e:
        log.error(f"[Shard {digit}] Alt failed: {e}")

    with load_lock:
        loaded_shards.add(digit)


def background_loader():
    """Load all 10 shards in background at startup."""
    log.info("[Loader] Background shard loading started...")
    start = time.time()
    for d in SHARDS:
        load_shard(d)
    log.info(f"[Loader] All shards loaded in {time.time() - start:.2f}s")


@app.on_event("startup")
async def startup_event():
    # Non-blocking: shards load in background so Render doesn't kill us
    threading.Thread(target=background_loader, daemon=True).start()
    log.info("[Startup] Background loader thread started")


# ---------- Exception Handler ----------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={
            "status": "rejected",
            "message": "Invalid endpoint. STRICTLY use /FetchData?Number=XXXXXXXXXX",
            "Developer": "@Maybechx"
        })
    return JSONResponse(status_code=exc.status_code, content={
        "detail": exc.detail, "Developer": "@Maybechx"
    })


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def root_landing_page():
    return HTMLResponse(content=LANDING_PAGE_HTML, status_code=200)


@app.get("/health")
def health():
    with load_lock:
        return {
            "status": "ok",
            "loaded_shards": sorted(list(loaded_shards)),
            "total_shards": len(SHARDS),
            "ready": len(loaded_shards) == len(SHARDS),
            "Developer": "@Maybechx"
        }


@app.get("/FetchData")
def fetch_data(Number: str = Query(None)):
    # ---- Validation ----
    if not Number or not Number.isdigit() or not (10 <= len(Number) <= 15):
        return JSONResponse(status_code=400, content={
            "status": "rejected",
            "message": "Invalid parameter. STRICTLY use /FetchData?Number=XXXXXXXXXX",
            "Developer": "@Maybechx"
        })

    last_digit = Number[-1]

    with load_lock:
        shard_ready = last_digit in loaded_shards

    # If shard not yet loaded (still warming up), fallback to direct parquet read
    if shard_ready:
        try:
            main_records = con.execute(
                f"SELECT * FROM main_shard_{last_digit} WHERE mobile = ?", [Number]
            ).df().to_dict(orient="records")
        except Exception as e:
            log.error(f"[Query main_shard_{last_digit}] {e}")
            main_records = []

        try:
            alt_records = con.execute(
                f"SELECT * FROM alt_shard_{last_digit} WHERE alt = ?", [Number]
            ).df().to_dict(orient="records")
        except Exception as e:
            log.error(f"[Query alt_shard_{last_digit}] {e}")
            alt_records = []
    else:
        # Fallback: direct parquet read from HuggingFace
        log.warning(f"[Shard {last_digit}] Not loaded yet — reading directly from HF")
        main_url = f"{HF_BASE}/final_master_shard_{last_digit}.parquet"
        alt_url = f"{HF_BASE}/alt_master_shard_{last_digit}.parquet"

        try:
            main_records = con.execute(
                "SELECT * FROM read_parquet(?) WHERE mobile = ?", [main_url, Number]
            ).df().to_dict(orient="records")
        except Exception as e:
            log.error(f"[Direct main] {e}")
            main_records = []

        try:
            alt_records = con.execute(
                "SELECT * FROM read_parquet(?) WHERE alt = ?", [alt_url, Number]
            ).df().to_dict(orient="records")
        except Exception as e:
            log.error(f"[Direct alt] {e}")
            alt_records = []

    if not main_records and not alt_records:
        return JSONResponse(status_code=404, content={
            "status": "not_found",
            "phone": Number,
            "Developer": "@Maybechx"
        })

    return {
        "status": "success",
        "Data": {
            "Main_Records": main_records,
            "Alt_Records": alt_records
        },
        "Developer": "@Maybechx"
    }


# ---------- Local Run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
