"""FastAPI + onnxruntime inference server untuk deteksi APD (Safety Helmet/Vest/Boot) (YOLOv8n)."""

import io
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# KONFIGURASI
# ============================================================

MODEL_PATH = Path(__file__).parent / "model.onnx"

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

CLASS_NAMES = ["Safety Helmet", "Safety Vest", "Safety Boot"]


# ============================================================
# DATABASE (MySQL via Railway)
# ============================================================
# Railway plugin MySQL otomatis menyediakan variabel MYSQL_URL / MYSQL_PUBLIC_URL.
# Di service backend, tambahkan environment variable DATABASE_URL yang me-reference
# variabel itu (Railway: New Variable -> Reference -> pilih service MySQL -> MYSQL_URL).
# Kalau DATABASE_URL tidak diset (misal dijalankan lokal tanpa DB), endpoint
# /checklist/* akan mengembalikan error yang jelas, tapi /predict tetap jalan normal.

RAW_DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("MYSQL_URL")

engine = None
SessionLocal = None
Base = declarative_base()

if RAW_DATABASE_URL:
    db_url = RAW_DATABASE_URL
    if db_url.startswith("mysql://"):
        db_url = db_url.replace("mysql://", "mysql+pymysql://", 1)
    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=280)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class ChecklistEntry(Base):
    __tablename__ = "checklist_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    technician = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)

    helmet = Column(Boolean, default=False)
    vest = Column(Boolean, default=False)
    shoes = Column(Boolean, default=False)

    conf_helmet = Column(Float, default=0.0)
    conf_vest = Column(Float, default=0.0)
    conf_shoes = Column(Float, default=0.0)

    approved = Column(Boolean, default=False)

    # Foto hasil capture, disimpan sebagai base64 data URL (mis. "data:image/jpeg;base64,...")
    image = Column(Text, nullable=True)


def get_db():
    if SessionLocal is None:
        raise HTTPException(
            status_code=503,
            detail="Database belum dikonfigurasi. Set environment variable DATABASE_URL di Railway.",
        )
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# LOAD MODEL ONNX
# ============================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PPE Detection API"
)


# ============================================================
# CORS
# ============================================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    if engine is not None:
        Base.metadata.create_all(bind=engine)


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(
    im,
    new_shape=IMG_SIZE,
    color=(114, 114, 114)
):
    shape = im.shape[:2]  # (height, width)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(
        new_shape[0] / shape[0],
        new_shape[1] / shape[1]
    )

    new_unpad = (
        int(round(shape[1] * r)),
        int(round(shape[0] * r))
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    im_resized = cv2.resize(
        im,
        new_unpad,
        interpolation=cv2.INTER_LINEAR
    )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im_padded = cv2.copyMakeBorder(
        im_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return im_padded, r, (left, top)


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(img_bgr):

    img, r, (dw, dh) = letterbox(
        img_bgr,
        IMG_SIZE
    )

    # BGR -> RGB
    img = img[:, :, ::-1]

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    # uint8 -> float32
    img = np.ascontiguousarray(
        img,
        dtype=np.float32
    ) / 255.0

    # Tambahkan batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    return img, r, dw, dh


# ============================================================
# POSTPROCESS
# ============================================================

def postprocess(
    outputs,
    r,
    dw,
    dh,
    orig_shape
):

    pred = np.squeeze(
        outputs[0]
    ).T

    # (num_boxes, 4 + jumlah kelas)
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]

    class_ids = np.argmax(
        scores_all,
        axis=1
    )

    confidences = np.max(
        scores_all,
        axis=1
    )

    # Filter confidence
    mask = confidences > CONF_THRES

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return []

    # ========================================================
    # xywh -> xyxy
    # ========================================================

    boxes_xyxy = np.zeros_like(
        boxes_xywh
    )

    boxes_xyxy[:, 0] = (
        boxes_xywh[:, 0]
        - boxes_xywh[:, 2] / 2
    )

    boxes_xyxy[:, 1] = (
        boxes_xywh[:, 1]
        - boxes_xywh[:, 3] / 2
    )

    boxes_xyxy[:, 2] = (
        boxes_xywh[:, 0]
        + boxes_xywh[:, 2] / 2
    )

    boxes_xyxy[:, 3] = (
        boxes_xywh[:, 1]
        + boxes_xywh[:, 3] / 2
    )

    # ========================================================
    # Kembalikan koordinat ke gambar asli
    # ========================================================

    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh

    boxes_xyxy /= r

    # ========================================================
    # Batasi bounding box
    # ========================================================

    h, w = orig_shape

    boxes_xyxy[:, [0, 2]] = boxes_xyxy[
        :, [0, 2]
    ].clip(0, w)

    boxes_xyxy[:, [1, 3]] = boxes_xyxy[
        :, [1, 3]
    ].clip(0, h)

    # ========================================================
    # NMS
    # ========================================================

    boxes_for_nms = boxes_xyxy.copy()

    boxes_for_nms[:, 2] -= boxes_for_nms[:, 0]
    boxes_for_nms[:, 3] -= boxes_for_nms[:, 1]

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(),
        confidences.tolist(),
        CONF_THRES,
        IOU_THRES
    )

    detections = []

    if len(indices) > 0:

        for i in np.array(indices).flatten():

            x1, y1, x2, y2 = boxes_xyxy[i]

            cls_id = int(
                class_ids[i]
            )

            detections.append({
                "class_id": cls_id,
                "class_name": (
                    CLASS_NAMES[cls_id]
                    if cls_id < len(CLASS_NAMES)
                    else str(cls_id)
                ),
                "confidence": float(
                    confidences[i]
                ),
                "box_xyxy": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            })

    return detections


# ============================================================
# API ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "PPE Detection API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    npimg = np.frombuffer(
        contents,
        np.uint8
    )

    img_bgr = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )

    if img_bgr is None:

        return JSONResponse(
            status_code=400,
            content={
                "error": "File gambar tidak valid"
            }
        )

    input_tensor, r, dw, dh = preprocess(
        img_bgr
    )

    outputs = session.run(
        None,
        {INPUT_NAME: input_tensor}
    )

    detections = postprocess(
        outputs,
        r,
        dw,
        dh,
        img_bgr.shape[:2]
    )

    return {
        "count": len(detections),
        "detections": detections
    }


# ============================================================
# CHECKLIST ENDPOINTS (MySQL)
# ============================================================

class ChecklistIn(BaseModel):
    technician: str
    location: str
    helmet: bool
    vest: bool
    shoes: bool
    conf_helmet: float = 0.0
    conf_vest: float = 0.0
    conf_shoes: float = 0.0
    image: str | None = None


def entry_to_dict(entry: "ChecklistEntry") -> dict:
    return {
        "id": f"CHK-{entry.id:04d}",
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "technician": entry.technician,
        "location": entry.location,
        "result": {
            "helmet": bool(entry.helmet),
            "vest": bool(entry.vest),
            "shoes": bool(entry.shoes),
        },
        "confidences": {
            "helmet": entry.conf_helmet,
            "vest": entry.conf_vest,
            "shoes": entry.conf_shoes,
        },
        "approved": bool(entry.approved),
        "image": entry.image,
    }


@app.post("/checklist")
def create_checklist(payload: ChecklistIn):
    if SessionLocal is None:
        raise HTTPException(
            status_code=503,
            detail="Database belum dikonfigurasi. Set environment variable DATABASE_URL di Railway.",
        )

    approved = payload.helmet and payload.vest and payload.shoes

    db = SessionLocal()
    try:
        entry = ChecklistEntry(
            timestamp=datetime.now(timezone.utc),
            technician=payload.technician,
            location=payload.location,
            helmet=payload.helmet,
            vest=payload.vest,
            shoes=payload.shoes,
            conf_helmet=payload.conf_helmet,
            conf_vest=payload.conf_vest,
            conf_shoes=payload.conf_shoes,
            approved=approved,
            image=payload.image,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry_to_dict(entry)
    finally:
        db.close()


@app.get("/checklist/history")
def get_history(limit: int = 200):
    if SessionLocal is None:
        raise HTTPException(
            status_code=503,
            detail="Database belum dikonfigurasi. Set environment variable DATABASE_URL di Railway.",
        )

    db = SessionLocal()
    try:
        entries = (
            db.query(ChecklistEntry)
            .order_by(ChecklistEntry.timestamp.desc())
            .limit(limit)
            .all()
        )
        return {"entries": [entry_to_dict(e) for e in entries]}
    finally:
        db.close()


DAY_NAMES_ID = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]


@app.get("/checklist/stats")
def get_stats():
    if SessionLocal is None:
        raise HTTPException(
            status_code=503,
            detail="Database belum dikonfigurasi. Set environment variable DATABASE_URL di Railway.",
        )

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        since_14d = now - timedelta(days=14)

        all_entries = (
            db.query(ChecklistEntry)
            .filter(ChecklistEntry.timestamp >= since_14d)
            .order_by(ChecklistEntry.timestamp.asc())
            .all()
        )

        total = db.query(ChecklistEntry).count()
        approved = db.query(ChecklistEntry).filter(ChecklistEntry.approved == True).count()  # noqa: E712
        rejected = total - approved

        all_confidences = []
        for e in all_entries:
            for c in (e.conf_helmet, e.conf_vest, e.conf_shoes):
                if c:
                    all_confidences.append(c)
        avg_confidence = round(sum(all_confidences) / len(all_confidences), 1) if all_confidences else 0

        # ---- Weekly (7 hari terakhir) ----
        since_7d = now - timedelta(days=7)
        weekly_buckets = {}
        for e in all_entries:
            if e.timestamp < since_7d:
                continue
            day_label = DAY_NAMES_ID[e.timestamp.weekday()]
            weekly_buckets.setdefault(day_label, {"approved": 0, "rejected": 0})
            if e.approved:
                weekly_buckets[day_label]["approved"] += 1
            else:
                weekly_buckets[day_label]["rejected"] += 1
        weekly = [
            {"day": d, **weekly_buckets.get(d, {"approved": 0, "rejected": 0})}
            for d in DAY_NAMES_ID
        ]

        # ---- Tingkat pemakaian tiap APD ----
        if total > 0:
            all_time_entries = db.query(ChecklistEntry).all()
            helmet_rate = round(100 * sum(1 for e in all_time_entries if e.helmet) / total, 1)
            vest_rate = round(100 * sum(1 for e in all_time_entries if e.vest) / total, 1)
            shoes_rate = round(100 * sum(1 for e in all_time_entries if e.shoes) / total, 1)
        else:
            helmet_rate = vest_rate = shoes_rate = 0
        apd_rates = [
            {"name": "Helm", "value": helmet_rate},
            {"name": "Vest", "value": vest_rate},
            {"name": "Sepatu", "value": shoes_rate},
        ]

        # ---- Trend 14 hari ----
        trend_buckets = {}
        for e in all_entries:
            date_label = e.timestamp.strftime("%d %b")
            trend_buckets.setdefault(date_label, {"approved": 0, "total": 0})
            trend_buckets[date_label]["total"] += 1
            if e.approved:
                trend_buckets[date_label]["approved"] += 1

        trend = []
        for i in range(13, -1, -1):
            d = now - timedelta(days=i)
            label = d.strftime("%d %b")
            bucket = trend_buckets.get(label)
            rate = round(100 * bucket["approved"] / bucket["total"]) if bucket else None
            trend.append({"date": label, "rate": rate})

        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "avg_confidence": avg_confidence,
            "weekly": weekly,
            "apd_rates": apd_rates,
            "trend": trend,
        }
    finally:
        db.close()

