"""V.S.P.M. live inference backend (FastAPI) - Render Free Tier.
No external database: bounded in-memory state guarded by a lock."""
import logging
import threading
import time
import warnings
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from scipy.stats import kurtosis

warnings.filterwarnings("ignore", message="X does not have valid feature names")
log = logging.getLogger("vspm")
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "vspm_rf_model.joblib"
SAMPLES = 800
BATCHES_PER_RUN = 100
IDLE_RESET_S = 15.0
LABELS = {0: "Healthy", 1: "Unbalance Warning", 2: "Critical Breakdown"}
DEFAULT_COLUMNS = ["x_rms", "x_p2p", "x_crest", "x_kurt", "y_rms", "y_p2p", "y_crest",
                   "y_kurt", "z_rms", "z_p2p", "z_crest", "z_kurt"]

MODEL = None
COLUMNS = DEFAULT_COLUMNS
LOCK = threading.Lock()
HISTORY = deque(maxlen=10)
RUN = None  # active/latest run dict, None when idle


# ---------------------------------------------------------------- model
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, COLUMNS
    try:
        bundle = joblib.load(MODEL_PATH)
        MODEL = bundle["model"]
        COLUMNS = list(bundle.get("feature_columns", DEFAULT_COLUMNS))
        if COLUMNS != DEFAULT_COLUMNS:
            log.warning("feature_columns differ from the expected 12-column order: %s", COLUMNS)
        log.info("Model loaded from %s (classes=%s)", MODEL_PATH.name, list(MODEL.classes_))
    except Exception as exc:  # missing file must not crash startup
        MODEL = None
        log.error("Could not load %s: %s. /predict will return 503.", MODEL_PATH, exc)
    yield


app = FastAPI(title="V.S.P.M. Inference", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errs = [{"field": ".".join(str(p) for p in e["loc"][1:]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"status": "error", "detail": errs[:5]})


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"status": "error", "detail": "internal error"})


# ---------------------------------------------------------------- schema
class Batch(BaseModel):
    x: List[float] = Field(min_length=SAMPLES, max_length=SAMPLES)
    y: List[float] = Field(min_length=SAMPLES, max_length=SAMPLES)
    z: List[float] = Field(min_length=SAMPLES, max_length=SAMPLES)
    batch_index: Optional[int] = None
    total_batches: int = Field(default=BATCHES_PER_RUN, ge=1, le=1000)
    run_id: Optional[str] = None
    remove_dc: bool = False
    is_final: Optional[bool] = None


# ---------------------------------------------------------------- features
def axis_features(sig, remove_dc):
    s = np.asarray(sig, dtype=np.float64)
    if remove_dc:
        s = s - s.mean()
    rms = float(np.sqrt(np.mean(s ** 2)))
    p2p = float(np.ptp(s))
    peak = float(np.max(np.abs(s)))
    crest = peak / rms if rms > 1e-12 else 0.0
    kurt = float(kurtosis(s, fisher=True))
    if not np.isfinite(kurt):
        kurt = 0.0
    return {"rms": rms, "p2p": p2p, "crest": crest, "kurt": kurt}


def extract(b: Batch):
    feats = {}
    for ax in ("x", "y", "z"):
        for k, v in axis_features(getattr(b, ax), b.remove_dc).items():
            feats[f"{ax}_{k}"] = v
    return feats


def infer(feats):
    row = np.array([[feats[c] for c in COLUMNS]], dtype=np.float64)
    pred = int(MODEL.predict(row)[0])
    raw = MODEL.predict_proba(row)[0]
    proba = [0.0, 0.0, 0.0]
    for cls, p in zip(MODEL.classes_, raw):
        if 0 <= int(cls) <= 2:
            proba[int(cls)] = float(p)
    return pred, proba


# ---------------------------------------------------------------- run state
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run(run_id, total):
    return {"run_id": run_id, "total": total, "batches": {}, "state": "collecting",
            "last_ts": time.monotonic(), "updated": now_iso(), "final": False}


def conclusion(run):
    """Run-level conclusion from majority vote + mean probability."""
    rows = [run["batches"][k] for k in sorted(run["batches"])]
    if not rows:
        return {"class_id": None, "label": "Awaiting Run", "confidence": 0.0, "finalized": False,
                "vote_breakdown": {"counts": [0, 0, 0], "percent": [0.0, 0.0, 0.0]},
                "mean_probabilities": [0.0, 0.0, 0.0], "majority_class_id": None,
                "timestamp": run["updated"] if run else None}
    counts = [0, 0, 0]
    for r in rows:
        counts[r["pred"]] += 1
    mean_p = np.mean([r["proba"] for r in rows], axis=0)
    cid = int(np.argmax(mean_p))
    maj = int(np.argmax(counts))
    return {"class_id": cid, "label": LABELS[cid], "confidence": float(mean_p[cid]),
            "finalized": run["state"] == "completed",
            "vote_breakdown": {"counts": counts, "percent": [100.0 * c / len(rows) for c in counts]},
            "mean_probabilities": [float(p) for p in mean_p], "majority_class_id": maj,
            "timestamp": run["updated"]}


def summary(run):
    rows = [run["batches"][k]["feats"] for k in sorted(run["batches"])] if run else []
    if not rows:
        return {"mean": {}, "max": {}}
    return {"mean": {c: float(np.mean([r[c] for r in rows])) for c in COLUMNS},
            "max": {c: float(np.max([r[c] for r in rows])) for c in COLUMNS}}


def telemetry(run):
    keys = sorted(run["batches"]) if run else []
    out = {"batch": list(range(1, len(keys) + 1)), "pred": [run["batches"][k]["pred"] for k in keys]}
    for c in COLUMNS:
        out[c] = [round(run["batches"][k]["feats"][c], 4) for k in keys]
    return out


def start_new(run, b, key, now):
    if run is None or run["state"] == "completed":
        return True
    if b.run_id is not None and b.run_id != run["run_id"]:
        return True
    if now - run["last_ts"] > IDLE_RESET_S:
        return True
    if b.batch_index in (0, 1) and run["batches"] and not (len(run["batches"]) == 1 and key in run["batches"]):
        return True
    return False


# ---------------------------------------------------------------- endpoints
@app.post("/predict")
def predict(b: Batch):
    global RUN
    if MODEL is None:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "model not loaded"})
    feats = extract(b)
    pred, proba = infer(feats)
    now = time.monotonic()
    with LOCK:
        key = b.batch_index if b.batch_index is not None else (len(RUN["batches"]) + 1 if RUN else 1)
        if start_new(RUN, b, key, now):
            RUN = new_run(b.run_id, b.total_batches)
            if b.batch_index is None:
                key = 1
        RUN["batches"][key] = {"pred": pred, "proba": proba, "feats": feats}
        RUN["last_ts"], RUN["updated"] = now, now_iso()
        RUN["total"] = b.total_batches
        received = len(RUN["batches"])
        if received >= RUN["total"] or b.is_final:
            if RUN["state"] != "completed":
                RUN["state"] = "completed"
                c = conclusion(RUN)
                HISTORY.appendleft({"run_id": RUN["run_id"], "timestamp": RUN["updated"],
                                    "batches": received, "class_id": c["class_id"],
                                    "label": c["label"], "confidence": c["confidence"]})
        c = conclusion(RUN)
        complete = RUN["state"] == "completed"
        total = RUN["total"]
    return {"status": "ok", "batches_received": received, "total_batches": total,
            "run_complete": complete, "batch_prediction": pred,
            "run_conclusion": c["label"], "confidence": round(c["confidence"], 4)}


@app.get("/latest-status")
def latest_status():
    with LOCK:
        run = RUN
        if run is None:
            state, rec, total, idle, rid = "idle", 0, BATCHES_PER_RUN, None, None
        else:
            state, rec, total = run["state"], len(run["batches"]), run["total"]
            idle, rid = round(time.monotonic() - run["last_ts"], 1), run["run_id"]
        return {"model_loaded": MODEL is not None, "updated": now_iso(),
                "run": {"state": state, "batches_received": rec, "total_batches": total,
                        "idle_seconds": idle, "run_id": rid},
                "conclusion": conclusion(run) if run else conclusion({"batches": {}, "updated": None, "state": "idle"}),
                "labels": [LABELS[i] for i in range(3)],
                "telemetry": telemetry(run), "summary": summary(run), "history": list(HISTORY)}


@app.post("/reset")
def reset():
    global RUN
    with LOCK:
        RUN = None
    return {"status": "ok", "state": "idle"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/")
def index():
    return FileResponse(BASE / "index.html", media_type="text/html")
