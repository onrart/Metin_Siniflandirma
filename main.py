import os
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import torch
from transformers import pipeline, AutoConfig

MODEL_PATH = os.getenv("MODEL_PATH", "./final_model")
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

app = FastAPI(title="HF Text Classifier Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic şemaları --------------------------------------------------------
class PredictRequest(BaseModel):
    texts: List[str] = Field(..., description="Sınıflandırılacak metin listesi")
    top_k: Optional[int] = Field(
        1,
        ge=1,
        description="Döndürülecek en yüksek skorlu etiket sayısı (single-label)",
    )
    multi_label: Optional[bool] = Field(False, description="Multi-label ise True")
    threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Multi-label için skor eşiği"
    )
    max_length: Optional[int] = Field(512, ge=8, le=4096)
    truncation: Optional[bool] = Field(True)


class LabelScore(BaseModel):
    label: str
    score: float


class PredictResponseItem(BaseModel):
    text: str
    predictions: List[LabelScore]


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


# --- Model yükleme ------------------------------------------------------------
classifier = None
labels: List[str] = []
num_labels = 0
device_for_pipeline = -1


@app.on_event("startup")
def load_model():
    global classifier, labels, num_labels, device_for_pipeline

    # Cihaz seçimi (CUDA varsa GPU kullan)
    if torch.cuda.is_available():
        device_for_pipeline = DEVICE_INDEX  # örn: 0 -> cuda:0
    else:
        device_for_pipeline = -1  # CPU

    # Pipeline yükle
    try:
        classifier = pipeline(
            task="text-classification",
            model=MODEL_PATH,
            device=device_for_pipeline,
        )
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {e}")

    # Etiketleri config'ten oku
    try:
        cfg = AutoConfig.from_pretrained(MODEL_PATH)
        # id2label dict indeks sırasına göre etiketleri çıkar
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) > 0:
            # int key'lere göre sırala
            labels_sorted = [id2label[i] for i in sorted(map(int, id2label.keys()))]
            labels.extend(labels_sorted)
        else:
            # id2label yoksa pipeline üzerinden tahminle yakalarız
            labels.extend(sorted({}))
    except Exception:
        pass

    num_labels = len(labels) if labels else None

    # Soğuk başlangıç gecikmesini azaltmak için mini ısınma
    _ = classifier(["warmup"], top_k=1)


# --- Yardımcılar --------------------------------------------------------------
def _ensure_top_k(top_k: Optional[int], n_labels: Optional[int]) -> int:
    if top_k is None or top_k < 1:
        return 1
    if n_labels:
        return min(top_k, n_labels)
    return top_k


def _round_scores(items: List[Dict], ndigits: int = 6) -> List[Dict]:
    return [
        {"label": it["label"], "score": round(float(it["score"]), ndigits)}
        for it in items
    ]


# --- Endpoint'ler -------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": "cuda" if device_for_pipeline != -1 else "cpu",
        "model_path": MODEL_PATH,
    }


@app.get("/labels")
def get_labels():
    return {"labels": labels, "count": len(labels) if labels else None}


@app.get("/meta")
def meta():
    return {
        "model_path": MODEL_PATH,
        "device": "cuda" if device_for_pipeline != -1 else "cpu",
        "num_labels": num_labels,
        "labels": labels or None,
        "note": "multi_label için threshold parametresi kullanılabilir.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model hazır değil.")

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts boş olamaz.")

    common_args = {
        "truncation": req.truncation,
        "max_length": req.max_length,
        "function_to_apply": "sigmoid" if req.multi_label else "softmax",
    }

    results: List[PredictResponseItem] = []

    if req.multi_label:
        # Tüm skorları al, threshold üstünü filtrele
        raw = classifier(req.texts, return_all_scores=True, **common_args)
        for text, scores in zip(req.texts, raw):
            # scores -> [{'label': 'LABEL_0', 'score': 0.12}, ...]
            filtered = [s for s in scores if float(s["score"]) >= req.threshold]
            if not filtered:
                # eşik üstü yoksa en yüksek skoru tek başına döndür
                best = max(scores, key=lambda s: float(s["score"]))
                filtered = [best]
            # azalan sırada sırala
            filtered = sorted(filtered, key=lambda s: float(s["score"]), reverse=True)
            results.append(
                PredictResponseItem(text=text, predictions=_round_scores(filtered))
            )
    else:
        # Single-label, top_k kullanılabilir
        tk = _ensure_top_k(req.top_k, num_labels)
        raw = classifier(req.texts, top_k=tk, **common_args)
        # top_k=1 -> her bir öğe dict döner, top_k>1 -> list döner
        for text, out in zip(req.texts, raw):
            if isinstance(out, dict):
                preds = [out]
            else:
                preds = out
            preds = sorted(preds, key=lambda s: float(s["score"]), reverse=True)
            results.append(
                PredictResponseItem(text=text, predictions=_round_scores(preds))
            )

    return PredictResponse(results=results)
