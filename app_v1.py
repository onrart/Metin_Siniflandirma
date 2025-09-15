# app.py
import os
import io
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import snapshot_download

# ==========================
# .env (opsiyonel)
# ==========================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Metin Sınıflandırma (HF Pipeline)", layout="wide")

# ==========================
# Ortam değişkenleri
# ==========================
ENV_MODEL_PATH1 = os.getenv("MODEL_PATH1", "onrart/final_model")
ENV_MODEL_PATH2 = os.getenv("MODEL_PATH2", "onrart/bertuk_v1")
ENV_MODEL_PATH3 = os.getenv("MODEL_PATH3", "onrart/mdeberta_v1")
ENV_MODEL_REVISION = os.getenv("MODEL_REVISION", None)  # opsiyonel
ENV_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")       # streamlit için kritik değil, ileride API modunda kullanılabilir

def _parse_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s is not None and s != "" else None
    except Exception:
        return None

ENV_DEVICE_INDEX = _parse_int(os.getenv("DEVICE_INDEX", None))  # -1=CPU, None=Auto(GPU0), 0..N=GPU

# ==========================
# Yardımcılar
# ==========================
def available_devices() -> Tuple[List[str], List[Optional[int]]]:
    """UI için görünen isimler ve karşılık gelen index/sentinel listesi döndürür."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        names = ["Otomatik (CUDA varsa GPU)", "CPU"] + [
            f"GPU:{i} - {torch.cuda.get_device_name(i)}" for i in range(count)
        ]
        idxs: List[Optional[int]] = [None, -1] + list(range(count))
    else:
        names = ["CPU"]
        idxs = [-1]
    return names, idxs

def resolve_device(device_index: Optional[int]) -> int:
    """
    Geçerli bir pipeline cihaz indexi döndürür:
    - CUDA varsa ve device_index None (Auto) -> 0
    - device_index 0..cuda_count-1 ise -> kendisi
    - aksi halde -> -1 (CPU)
    """
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if device_index is None:  # Auto
            return 0
        if isinstance(device_index, int) and 0 <= device_index < cuda_count:
            return device_index
    return -1  # CPU fallback

def _is_local_path(path: str) -> bool:
    return os.path.isdir(path)

def _resolve_model_source(model_path: str, revision: Optional[str]) -> str:
    """
    HF repo id veya yerel klasör olabilir.
    - Yerel klasörse doğrudan onu döndürür.
    - Repo id ise:
        - revision varsa snapshot indir ve yerel yolu döndür (sabit sürüm).
        - yoksa repo id'yi döndür (transformers otomatik indir/cache'ler).
    """
    if _is_local_path(model_path):
        return model_path
    if revision:
        local_path = snapshot_download(repo_id=model_path, revision=revision)
        return local_path
    return model_path

@st.cache_resource(show_spinner=False)
def load_classifier(model_id_or_path: str, device_index: Optional[int], revision: Optional[str]):
    """
    Tokenizer + Model ayrı yüklenir (torch_dtype='auto') -> pipeline döndürür.
    Ayrıca (labels, device_str) döner.
    Cache: model_id_or_path, device_index, revision üçlüsüne göre.
    """
    device_for_pipeline = resolve_device(device_index)
    model_to_load = _resolve_model_source(model_id_or_path, revision)

    tok = AutoTokenizer.from_pretrained(model_to_load)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_to_load,
        torch_dtype="auto",  # CUDA'da fp16/bf16; CPU'da fp32
    )
    mdl.eval()

    clf = pipeline(
        task="text-classification",
        model=mdl,
        tokenizer=tok,
        device=device_for_pipeline,
    )

    # Etiketler
    labels: List[str] = []
    try:
        cfg = AutoConfig.from_pretrained(model_to_load)
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) > 0:
            labels = [id2label[i] for i in sorted(map(int, id2label.keys()))]
    except Exception:
        pass

    # Hafif ısınma (sadece GPU'da)
    try:
        if device_for_pipeline != -1:
            _ = clf(["warmup"], top_k=1, truncation=True, max_length=32)
    except Exception:
        pass

    device_str = "cuda" if device_for_pipeline != -1 else "cpu"
    return clf, labels, device_str

def predict_texts(
    clf,
    texts: List[str],
    multi_label: bool,
    threshold: float,
    top_k: int,
    max_length: int,
    truncation: bool,
    pipeline_batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Metin listesi için tahminleri döndürür."""
    if not texts:
        return []

    threshold = max(0.0, min(1.0, float(threshold)))
    top_k = max(1, int(top_k))
    max_length = max(8, int(max_length))
    pipeline_batch_size = max(1, int(pipeline_batch_size))

    tokenizer_kwargs = {
        "truncation": truncation,
        "max_length": max_length,
    }
    apply_fn = "sigmoid" if multi_label else "softmax"

    results: List[Dict[str, Any]] = []

    if multi_label:
        raw = clf(
            texts,
            return_all_scores=True,
            batch_size=pipeline_batch_size,
            function_to_apply=apply_fn,
            **tokenizer_kwargs,
        )
        for text, scores in zip(texts, raw):
            filtered = [s for s in scores if float(s["score"]) >= threshold]
            if not filtered:
                filtered = [max(scores, key=lambda s: float(s["score"]))]  # en az 1 etiket
            filtered.sort(key=lambda s: float(s["score"]), reverse=True)
            results.append(
                {
                    "text": text,
                    "predictions": [
                        {"label": s["label"], "score": round(float(s["score"]), 6)}
                        for s in filtered
                    ],
                }
            )
    else:
        raw = clf(
            texts,
            top_k=top_k,
            batch_size=pipeline_batch_size,
            function_to_apply=apply_fn,
            **tokenizer_kwargs,
        )
        for text, out in zip(texts, raw):
            preds = out if isinstance(out, list) else [out]
            preds.sort(key=lambda s: float(s["score"]), reverse=True)
            results.append(
                {
                    "text": text,
                    "predictions": [
                        {"label": p["label"], "score": round(float(p["score"]), 6)}
                        for p in preds
                    ],
                }
            )
    return results

def format_results_dataframe(results: List[Dict[str, Any]], multi_label: bool) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        text = r["text"]
        preds = r["predictions"]
        row = {
            "text": text,
            "predicted_labels": ", ".join([p["label"] for p in preds]),
            "predicted_scores": ", ".join([str(p["score"]) for p in preds]),
        }
        if not multi_label and preds:
            row["best_label"] = preds[0]["label"]
            row["best_score"] = preds[0]["score"]
        rows.append(row)
    return pd.DataFrame(rows)

def format_results_wide(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Her etiket için ayrı skor kolonu (yoksa 0.0) ekler."""
    if not results:
        return pd.DataFrame()
    all_labels = sorted({p["label"] for r in results for p in r["predictions"]})
    rows = []
    for r in results:
        row = {"text": r["text"]}
        for lbl in all_labels:
            row[f"score__{lbl}"] = 0.0
        for p in r["predictions"]:
            row[f"score__{p['label']}"] = float(p["score"])
        rows.append(row)
    return pd.DataFrame(rows)

def robust_read_csv(uploaded_file) -> pd.DataFrame:
    """Encoding/ayraç esnek CSV okuyucu."""
    raw = uploaded_file.read()
    # Basit bir yaklaşım: birkaç ayraç dener
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), engine="python")

# ==========================
# Tek-aktif model: session state
# ==========================
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "labels" not in st.session_state:
    st.session_state.labels = []
if "device_used" not in st.session_state:
    st.session_state.device_used = None
if "active_model_signature" not in st.session_state:
    st.session_state.active_model_signature = None  # (path, device_idx, revision)

def unload_current_model():
    """Cache ve GPU RAM boşalt."""
    try:
        st.session_state.classifier = None
        st.session_state.labels = []
        st.session_state.device_used = None
        st.session_state.active_model_signature = None
        try:
            load_classifier.clear()
        except Exception:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def load_active_model(path: str, device_idx: Optional[int], rev: Optional[str]):
    """Seçili modeli yükle; önce farklı bir model varsa boşalt."""
    sig = (path, device_idx, rev)
    if st.session_state.active_model_signature == sig and st.session_state.classifier is not None:
        return
    unload_current_model()
    clf, labels, device_used = load_classifier(path, device_idx, rev)
    st.session_state.classifier = clf
    st.session_state.labels = labels
    st.session_state.device_used = device_used
    st.session_state.active_model_signature = sig

# ==========================
# UI — Sidebar
# ==========================
st.sidebar.header("⚙️ Ayarlar")

# Hazır modeller (ENV'den)
PREDEFINED_MODELS = {
    "Final Model": ENV_MODEL_PATH1,
    "BERTurk v1": ENV_MODEL_PATH2,
    "mDeBERTa v1": ENV_MODEL_PATH3,
    "Custom (kendi yolum)": None,
}

# Cihaz seçimi
device_names, device_idxs = available_devices()

def _default_device_index_from_env() -> int:
    if torch.cuda.is_available():
        if ENV_DEVICE_INDEX is None:
            return 0  # Auto
        if ENV_DEVICE_INDEX == -1:
            return 1  # CPU
        try:
            return device_idxs.index(ENV_DEVICE_INDEX)
        except ValueError:
            return 0
    else:
        return 0

default_idx = _default_device_index_from_env()
device_choice = st.sidebar.selectbox("Cihaz", device_names, index=default_idx)
chosen_device_index: Optional[int] = device_idxs[device_names.index(device_choice)]

# Revision (opsiyonel)
revision = st.sidebar.text_input(
    "MODEL_REVISION (opsiyonel)",
    value=ENV_MODEL_REVISION or "",
    help="Örn: 'main' veya bir commit hash; boş bırakılırsa en son sürüm kullanılır",
)
revision = revision if revision.strip() else None

# Pipeline batch size
pipeline_batch_size = st.sidebar.number_input(
    "Pipeline batch size", min_value=1, max_value=128, value=32, step=1
)

# Model seçimi
model_choice = st.sidebar.selectbox(
    "Model seç",
    options=list(PREDEFINED_MODELS.keys()),
    index=0,
)

if PREDEFINED_MODELS[model_choice] is None:
    # Custom model
    model_path = st.sidebar.text_input(
        "Custom model yolu (HF repo id veya yerel klasör)",
        value="",
        help="Örn: onrart/final_model veya ./final_model",
    ).strip()
else:
    model_path = PREDEFINED_MODELS[model_choice]

# Yükle/Boşalt butonları
load_disabled = (not model_path) and (PREDEFINED_MODELS[model_choice] is None)
if st.sidebar.button("Modeli Yükle", type="primary", use_container_width=True, disabled=load_disabled):
    try:
        with st.spinner("Model yükleniyor..."):
            load_active_model(model_path, chosen_device_index, revision)
        st.sidebar.success(f"Model yüklendi · Cihaz: {st.session_state.device_used}")
        if revision:
            st.sidebar.caption(f"📌 Kullanılan sürüm: {revision}")
        if st.session_state.labels:
            st.sidebar.caption("Etiketler:")
            st.sidebar.write(st.session_state.labels)
    except Exception as e:
        st.sidebar.error(f"Model yüklenemedi: {e}")

if st.sidebar.button("Modeli Boşalt", use_container_width=True):
    unload_current_model()
    st.sidebar.info("Aktif model boşaltıldı.")

# Genel sınıflandırma ayarları
multi_label = st.sidebar.toggle("Multi-label", value=False, help="Birden fazla etiket aynı anda seçilebilir")
threshold = st.sidebar.slider("Eşik (multi-label)", 0.0, 1.0, 0.5, 0.01, disabled=not multi_label)
top_k = st.sidebar.number_input("Top-K (single-label)", min_value=1, max_value=10, value=3, step=1, disabled=multi_label)
max_length = st.sidebar.number_input("max_length", min_value=8, max_value=4096, value=512, step=8)
truncation = st.sidebar.toggle("Truncation", value=True)
csv_batch_size = st.sidebar.number_input("CSV batch size", min_value=8, max_value=4096, value=256, step=8)

# ==========================
# UI — İçerik
# ==========================
st.title("🧠 Metin Sınıflandırma — Streamlit (HF Pipeline)")

# Model yüklü mü?
if st.session_state.classifier is None:
    st.warning("Önce sol menüden modeli seçip **Modeli Yükle**'ye bas.")
    st.stop()

classifier = st.session_state.classifier
labels = st.session_state.labels
device_used = st.session_state.device_used

# top_k'yi etiket sayısına göre sınırla
if labels:
    max_top_k = max(1, min(10, len(labels)))
else:
    max_top_k = 10
if not multi_label:
    top_k = min(top_k, max_top_k)

tabs = st.tabs(["Tek Metin", "Çoklu Metin", "CSV Yükle"])

# ---- Tek Metin
with tabs[0]:
    st.subheader("Tek Metin Tahmini")
    text = st.text_area(
        "Metin",
        placeholder="Örn: kargo geç geldi ama müşteri hizmetleri iyiydi",
        height=140,
    )
    if st.button("Tahmin Et", type="primary", use_container_width=True, disabled=not text.strip()):
        with st.spinner("Tahmin yapılıyor..."):
            res = predict_texts(
                classifier,
                [text],
                multi_label=multi_label,
                threshold=threshold,
                top_k=top_k,
                max_length=max_length,
                truncation=truncation,
                pipeline_batch_size=pipeline_batch_size,
            )
        df = format_results_dataframe(res, multi_label=multi_label)
        st.dataframe(df, use_container_width=True)
        st.caption("Skorlar 0–1 aralığındadır.")
        if not multi_label and res and len(res[0]["predictions"]) > 1:
            chart_df = pd.DataFrame(res[0]["predictions"])
            st.bar_chart(chart_df.set_index("label")["score"], use_container_width=True)

# ---- Çoklu Metin
with tabs[1]:
    st.subheader("Çoklu Metin (her satır 1 metin)")
    bulk_texts = st.text_area(
        "Metinler",
        placeholder="Bir satıra bir metin olacak şekilde yapıştır.",
        height=220,
    )
    add_wide = st.checkbox("Wide format skor kolonlarını da ekle (score__etiket)", value=True)
    if st.button(
        "Toplu Tahmin Et",
        type="primary",
        use_container_width=True,
        disabled=not bulk_texts.strip(),
    ):
        texts = [t.strip() for t in bulk_texts.splitlines() if t.strip()]
        with st.spinner(f"{len(texts)} metin için tahmin yapılıyor..."):
            res = predict_texts(
                classifier,
                texts,
                multi_label=multi_label,
                threshold=threshold,
                top_k=top_k,
                max_length=max_length,
                truncation=truncation,
                pipeline_batch_size=pipeline_batch_size,
            )
        df_base = format_results_dataframe(res, multi_label=multi_label)
        if add_wide:
            df_wide = format_results_wide(res)
            df = df_base.merge(df_wide, on="text", how="left")
        else:
            df = df_base
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Sonuçları CSV indir",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

# ---- CSV Yükle
with tabs[2]:
    st.subheader("CSV Yükle ve Tahmin")
    up = st.file_uploader("CSV seç (utf-8, ; , \\t | desteklenir)", type=["csv"])
    text_col = st.text_input("Metin sütunu adı", value="text")
    add_wide_csv = st.checkbox("Wide format skor kolonlarını ekle (score__etiket)", value=True)
    if up is not None:
        try:
            df_in = robust_read_csv(up)
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")
            st.stop()
        st.write("Örnek ilk 5 satır:")
        st.dataframe(df_in.head(), use_container_width=True)
        if text_col not in df_in.columns:
            st.error(f"'{text_col}' sütunu bulunamadı.")
        else:
            if st.button("CSV'yi Tahmin Et", type="primary", use_container_width=True):
                texts = df_in[text_col].astype(str).tolist()
                results_all: List[Dict[str, Any]] = []
                with st.spinner(f"{len(texts)} satır işleniyor..."):
                    total = len(texts)
                    done = 0
                    prog = st.progress(0.0)
                    for i in range(0, total, int(csv_batch_size)):
                        chunk = texts[i : i + int(csv_batch_size)]
                        res = predict_texts(
                            classifier,
                            chunk,
                            multi_label=multi_label,
                            threshold=threshold,
                            top_k=top_k,
                            max_length=max_length,
                            truncation=truncation,
                            pipeline_batch_size=pipeline_batch_size,
                        )
                        results_all.extend(res)
                        done += len(chunk)
                        prog.progress(min(1.0, done / max(1, total)))
                df_out = format_results_dataframe(results_all, multi_label=multi_label)
                if add_wide_csv:
                    df_wide = format_results_wide(results_all)
                    df_final = pd.concat(
                        [
                            df_in.reset_index(drop=True),
                            df_out.drop(columns=["text"], errors="ignore").reset_index(drop=True),
                            df_wide.drop(columns=["text"], errors="ignore").reset_index(drop=True),
                        ],
                        axis=1,
                    )
                else:
                    df_final = pd.concat(
                        [
                            df_in.reset_index(drop=True),
                            df_out.drop(columns=["text"], errors="ignore").reset_index(drop=True),
                        ],
                        axis=1,
                    )
                st.success("Tamamlandı.")
                st.dataframe(df_final.head(50), use_container_width=True)
                csv = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Sonuçları CSV indir",
                    data=csv,
                    file_name="predictions_with_input.csv",
                    mime="text/csv",
                )

st.caption(
    "Notlar: 1) Bu uygulama tek-aktif model çalışır; model değiştirince önceki pipeline boşaltılır."
)
