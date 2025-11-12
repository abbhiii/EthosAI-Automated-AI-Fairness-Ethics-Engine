# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix
import uvicorn
import os
import sys
import subprocess
import numpy as np
import logging

# =========================
# Basic config & logging
# =========================
APP_NAME = "EthosAI - Fairness & Report API"
APP_VERSION = "0.3.0"
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20MB guardrail

# Allow overriding CORS via env (handy for AWS/GCP later)
FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ethosai")

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helpers
# =========================
def _ensure_csv_bytes(file: UploadFile) -> bytes:
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a CSV (.csv or .txt)")

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_FILE_BYTES//(1024*1024)}MB)")
    return data

def _read_csv(bytes_data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(bytes_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {e}")

def map_income_to01(series: pd.Series) -> pd.Series:
    """
    Robust mapping for Adult 'income' labels. Works regardless of quotes/spaces/punctuation.
    1 if string contains '>50k', else 0 if string contains '<=50k', else 0.
    """
    s = series.astype(str).str.lower().str.replace('"', '', regex=False).str.replace(' ', '', regex=False)
    is_pos = s.str.contains('>50k', regex=False)
    is_neg = s.str.contains('<=50k', regex=False)
    out = np.where(is_pos, 1, np.where(is_neg, 0, 0)).astype(int)
    return pd.Series(out, index=series.index)

def to01_generic(series: pd.Series) -> pd.Series:
    """Fallback mapper for arbitrary binary targets, using common positives."""
    s = series.astype(str).str.strip().str.lower().str.replace('"', '', regex=False)
    positives = {"1","true","yes","y","positive",">50k",">50k.","income>50k","gt50k","gt50k."}
    return s.apply(lambda x: 1 if (x in positives or '>50k' in x) else 0).astype(int)

def _maybe_numeric_threshold(df: pd.DataFrame, col: str, y_num: pd.Series) -> Tuple[pd.Series, Optional[float]]:
    """
    If mapping produced all zeros or all ones and target appears continuous,
    fallback to median threshold.
    """
    threshold_used = None
    if (y_num.sum() == 0 or y_num.sum() == len(y_num)):
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notnull().sum() > 0 and coerced.nunique() > 2:
            thresh = float(coerced.median())
            y_num = (coerced > thresh).fillna(0).astype(int)
            threshold_used = thresh
    return y_num, threshold_used

def compute_basic_fairness(df: pd.DataFrame, target_col: str, sensitive_col: str):
    """
    Minimal fairness report:
      - overall class distribution
      - positive rate by sensitive group (demographic parity)
      - (TPR/FPR filled later if predictions provided)
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in dataset")
    if sensitive_col not in df.columns:
        raise ValueError(f"sensitive_col '{sensitive_col}' not in dataset")

    df = df.copy()

    # Try robust income mapping first, then generic, then numeric threshold
    y_str = df[target_col].astype(str)
    y_num = map_income_to01(y_str)
    if y_num.sum() == 0 and (y_num == 0).all():
        y_num = to01_generic(y_str)
    y_num, threshold_used = _maybe_numeric_threshold(df, target_col, y_num)

    df["_target_bin"] = y_num
    groups = df[sensitive_col].astype(str).unique().tolist()

    report = {
        "overall": {
            "n_rows": int(len(df)),
            "positive_rate": float(df["_target_bin"].mean()),
            "class_counts": {str(k): int(v) for k, v in df["_target_bin"].value_counts().to_dict().items()}
        },
        "by_group": {}
    }

    for g in sorted(groups):
        sub = df[df[sensitive_col].astype(str) == g]
        if len(sub) == 0:
            continue
        pos_rate = float(sub["_target_bin"].mean())
        report["by_group"][g] = {
            "n": int(len(sub)),
            "positive_rate": pos_rate,
            "tpr": None,
            "fpr": None
        }

    debug = {"threshold_used": threshold_used}
    return report, debug

def _suggest_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Heuristically suggest:
      - targets: likely label columns (binary / names like 'label', 'target', 'income', 'y')
      - protected: columns like 'sex', 'gender', 'race', 'age', 'marital-status', etc. (categorical, few unique)
    """
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    likely_targets = []
    for c in cols:
        lc = lower[c]
        if any(k in lc for k in ["target", "label", "y", "income", "clicked", "churn"]):
            likely_targets.append(c)

    # also include binary columns by unique count
    for c in cols:
        try:
            nunq = df[c].nunique(dropna=True)
            if nunq == 2 and c not in likely_targets:
                likely_targets.append(c)
        except Exception:
            pass

    likely_protected = []
    protected_keywords = [
        "sex","gender","race","ethnicity","age","nationality","marital","religion","skin","disability","caste"
    ]
    for c in cols:
        lc = lower[c]
        if any(k in lc for k in protected_keywords):
            likely_protected.append(c)

    # also include low-cardinality categoricals (e.g., <= 10 unique)
    for c in cols:
        try:
            nunq = df[c].nunique(dropna=True)
            if nunq > 1 and nunq <= 10 and c not in likely_protected:
                likely_protected.append(c)
        except Exception:
            pass

    # keep order stable & unique
    def _uniq(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return _uniq(likely_targets)[:10], _uniq(likely_protected)[:10]

# =========================
# Schemas (for clarity)
# =========================
class AnalyzeResponse(BaseModel):
    status: str
    analysis: Dict[str, Any]

# =========================
# Routes
# =========================
@app.get("/", response_class=PlainTextResponse)
def root():
    return f"{APP_NAME} v{APP_VERSION}"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/analyze-dataset", response_model=AnalyzeResponse)
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Quick column analysis for UI:
      - returns column names + dtypes + unique counts
      - suggests likely target & protected columns
    """
    data = _ensure_csv_bytes(file)
    df = _read_csv(data)

    cols_info = {}
    for c in df.columns:
        try:
            cols_info[c] = {
                "dtype": str(df[c].dtype),
                "nunique": int(df[c].nunique(dropna=True)),
                "example": None if df[c].dropna().empty else str(df[c].dropna().iloc[0])[:120],
            }
        except Exception:
            cols_info[c] = {"dtype": "unknown", "nunique": -1, "example": None}

    suggested_targets, suggested_protected = _suggest_columns(df)

    return {
        "status": "ok",
        "analysis": {
            "columns": cols_info,
            "suggested_targets": suggested_targets,
            "suggested_protected": suggested_protected,
        }
    }

@app.post("/api/upload-dataset")
async def upload_dataset(
    dataset: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    predictions_col: Optional[str] = Form(None)
):
    """
    Upload a CSV dataset and compute a minimal fairness report.

    Form fields:
    - dataset: CSV file
    - target_col: name of the label/target column (binary)
    - sensitive_col: sensitive attribute column (e.g., sex, race)
    - predictions_col (optional): if provided, computes TPR/FPR per group.
    """
    data = _ensure_csv_bytes(dataset)
    df = _read_csv(data)

    # Base report with robust mapping
    try:
        report, mapping_debug = compute_basic_fairness(df, target_col, sensitive_col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # If predictions provided, compute TPR/FPR per group
    if predictions_col:
        if predictions_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"predictions_col '{predictions_col}' not in dataset")

        # map true/pred labels robustly
        y_true = map_income_to01(df[target_col])
        if y_true.sum() == 0 and y_true.mean() == 0:
            y_true = to01_generic(df[target_col])

        y_pred = to01_generic(df[predictions_col])

        sens = df[sensitive_col].astype(str).values
        groups = list(report["by_group"].keys())

        for g in groups:
            mask = (sens == g)
            if mask.sum() == 0:
                continue
            yt = y_true[mask].values
            yp = y_pred[mask].values
            # Safe confusion matrix
            tn = fp = fn = tp = 0
            try:
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            except Exception:
                for a, b in zip(yt, yp):
                    if a == 0 and b == 0: tn += 1
                    elif a == 0 and b == 1: fp += 1
                    elif a == 1 and b == 0: fn += 1
                    elif a == 1 and b == 1: tp += 1

            tpr = (tp / (tp + fn)) if (tp + fn) > 0 else None
            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else None
            report["by_group"][g]["tpr"] = tpr
            report["by_group"][g]["fpr"] = fpr

    # Debug info to verify mapping (and show a few unique raw labels)
    _s = df[target_col].astype(str).str.lower().str.replace('"','', regex=False).str.replace(' ','', regex=False)
    _pos = int(_s.str.contains('>50k', regex=False).sum())
    _neg = int(_s.str.contains('<=50k', regex=False).sum())
    _uniq = _s.unique().tolist()[:6]
    debug = {"unique": _uniq, "pos": _pos, "neg": _neg}
    if isinstance(mapping_debug, dict):
        debug.update(mapping_debug)

    return JSONResponse(content={"status": "ok", "report": report, "debug": debug})

@app.post("/api/generate-report")
def generate_report():
    """
    Run llm_report.py (recomputes metrics on YOUR local dataset at ../data/adult.csv)
    and return the report text as JSON.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
    script_path = os.path.join(BASE_DIR, "llm_report.py")
    report_path = os.path.join(BASE_DIR, "ethos_ai_report.txt")

    if not os.path.exists(script_path):
        return {"status": "error", "message": f"Script not found: {script_path}"}

    try:
        _ = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        return {"status": "error", "stderr": e.stderr, "stdout": e.stdout}

    if not os.path.exists(report_path):
        return {"status": "error", "message": "Report file not found after generation."}

    with open(report_path, "r") as f:
        text = f.read()

    return {"status": "ok", "report": text}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=True)
