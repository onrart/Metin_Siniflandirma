# app.py
import os
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
import torch
from transformers import pipeline, AutoConfig
from huggingface_hub import snapshot_download

# .env (opsiyonel)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Metin Sınıflandırma (HF Pipeline)", layout="wide")

# ==========================
# Ortam değişkenleri (varsayılanlar)
# ==========================
ENV_MODEL_PATH = os.getenv(
    "MODEL_PATH", "onrart/final_model"
)  # <— varsayılan HF repo id
ENV_MODEL_REVISION = os.getenv(
    "MODEL_REVISION", None
)  # opsiyonel: "main" veya commit hash


def _parse_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s is not None and s != "" else None
    except Exception:
        return None


ENV_DEVICE_INDEX = _parse_int(
    os.getenv("DEVICE_INDEX", None)
)  # -1=CPU, None=Auto(GPU0), 0..N=GPU


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
    # Basit kontrol: klasör mevcutsa yerel kabul et
    return os.path.isdir(path)


def _resolve_model_source(model_path: str) -> str:
    """
    HF repo id veya yerel klasör olabilir.
    - Yerel klasörse doğrudan onu döndürür.
    - Repo id ise:
        - MODEL_REVISION varsa snapshot indirir ve yerel yolu döndürür (sabitlenmiş sürüm).
        - Yoksa repo id'yi döndürür (transformers otomatik indirir/cache'ler).
    """
    if _is_local_path(model_path):
        return model_path
    # HF repo id olduğunu varsay
    if ENV_MODEL_REVISION:
        local_path = snapshot_download(repo_id=model_path, revision=ENV_MODEL_REVISION)
        return local_path
    return model_path


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str, device_index: Optional[int]):
    """
    HF pipeline'ı cache'leyerek yükler ve (classifier, labels, device_str) döner.
    """
    device_for_pipeline = resolve_device(device_index)
    model_to_load = _resolve_model_source(model_path)

    # Pipeline yükle
    clf = pipeline(
        task="text-classification",
        model=model_to_load,
        device=device_for_pipeline,
    )

    # Etiketleri config'ten çek (varsa)
    labels: List[str] = []
    try:
        cfg = AutoConfig.from_pretrained(model_to_load)
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) > 0:
            labels = [id2label[i] for i in sorted(map(int, id2label.keys()))]
    except Exception:
        pass

    # Isınma (soğuk başlangıcı azaltır)
    try:
        _ = clf(["warmup"], top_k=1)
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
) -> List[Dict[str, Any]]:
    """Metin listesi için tahminleri döndürür."""
    if not texts:
        return []

    threshold = max(0.0, min(1.0, float(threshold)))
    top_k = max(1, int(top_k))
    max_length = max(8, int(max_length))

    common_args = {
        "truncation": truncation,
        "max_length": max_length,
        "function_to_apply": "sigmoid" if multi_label else "softmax",
    }

    results: List[Dict[str, Any]] = []

    if multi_label:
        raw = clf(texts, return_all_scores=True, **common_args)
        for text, scores in zip(texts, raw):
            filtered = [s for s in scores if float(s["score"]) >= threshold]
            if not filtered:
                filtered = [max(scores, key=lambda s: float(s["score"]))]
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
        raw = clf(texts, top_k=top_k, **common_args)
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


def format_results_dataframe(
    results: List[Dict[str, Any]], multi_label: bool
) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        text = r["text"]
        preds = r["predictions"]
        if multi_label:
            rows.append(
                {
                    "text": text,
                    "predicted_labels": ", ".join([p["label"] for p in preds]),
                    "predicted_scores": ", ".join([str(p["score"]) for p in preds]),
                }
            )
        else:
            rows.append(
                {
                    "text": text,
                    "predicted_labels": ", ".join([p["label"] for p in preds]),
                    "predicted_scores": ", ".join([str(p["score"]) for p in preds]),
                    "best_label": preds[0]["label"],
                    "best_score": preds[0]["score"],
                }
            )
    return pd.DataFrame(rows)


# ==========================
# UI
# ==========================
st.sidebar.header("⚙️ Ayarlar")

# Model yolu (ENV varsayılanıyla başlar)
model_path = st.sidebar.text_input(
    "Model yolu (HF repo id veya yerel klasör)",
    value=ENV_MODEL_PATH,
    help="Örn: onrart/final_model veya ./final_model",
)

# Cihaz seçimi: ENV_DEVICE_INDEX'e göre başlangıç seçimi
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

multi_label = st.sidebar.toggle(
    "Multi-label", value=False, help="Birden fazla etiket aynı anda seçilebilir"
)
threshold = st.sidebar.slider(
    "Eşik (multi-label)", 0.0, 1.0, 0.5, 0.01, disabled=not multi_label
)
top_k = st.sidebar.number_input(
    "Top-K (single-label)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    disabled=multi_label,
)
max_length = st.sidebar.number_input(
    "max_length", min_value=8, max_value=4096, value=512, step=8
)
truncation = st.sidebar.toggle("Truncation", value=True)
batch_size = st.sidebar.number_input(
    "CSV batch size", min_value=8, max_value=4096, value=256, step=8
)

# Modeli yükle
with st.sidebar:
    try:
        with st.spinner("Model yükleniyor..."):
            classifier, labels, device_used = load_classifier(
                model_path, chosen_device_index
            )
        st.success(f"Model yüklendi · Cihaz: {device_used}")
        if ENV_MODEL_REVISION:
            st.caption(f"📌 Kullanılan sürüm: {ENV_MODEL_REVISION}")
        if labels:
            st.caption("Etiketler:")
            st.write(labels)
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        st.stop()

st.title("🧠 Metin Sınıflandırma — Streamlit (HF Pipeline)")

tabs = st.tabs(["Tek Metin", "Çoklu Metin", "CSV Yükle"])

# ---- Tek Metin
with tabs[0]:
    st.subheader("Tek Metin Tahmini")
    text = st.text_area(
        "Metin",
        placeholder="Örn: kargo geç geldi ama müşteri hizmetleri iyiydi",
        height=140,
    )
    if st.button(
        "Tahmin Et", type="primary", use_container_width=True, disabled=not text.strip()
    ):
        with st.spinner("Tahmin yapılıyor..."):
            res = predict_texts(
                classifier,
                [text],
                multi_label=multi_label,
                threshold=threshold,
                top_k=top_k,
                max_length=max_length,
                truncation=truncation,
            )
        df = format_results_dataframe(res, multi_label=multi_label)
        st.dataframe(df, use_container_width=True)
        if not multi_label and res and len(res[0]["predictions"]) > 1:
            chart_df = pd.DataFrame(res[0]["predictions"])
            st.bar_chart(chart_df.set_index("label")["score"])

# ---- Çoklu Metin
with tabs[1]:
    st.subheader("Çoklu Metin (her satır 1 metin)")
    bulk_texts = st.text_area(
        "Metinler",
        placeholder="Bir satıra bir metin olacak şekilde yapıştır.",
        height=220,
    )
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
            )
        df = format_results_dataframe(res, multi_label=multi_label)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Sonuçları CSV indir",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

# ---- CSV Yükle
with tabs[2]:
    st.subheader("CSV Yükle ve Tahmin")
    up = st.file_uploader("CSV seç (utf-8)", type=["csv"])
    text_col = st.text_input("Metin sütunu adı", value="text")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
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
                    for i in range(0, total, int(batch_size)):
                        chunk = texts[i : i + int(batch_size)]
                        res = predict_texts(
                            classifier,
                            chunk,
                            multi_label=multi_label,
                            threshold=threshold,
                            top_k=top_k,
                            max_length=max_length,
                            truncation=truncation,
                        )
                        results_all.extend(res)
                        done += len(chunk)
                        prog.progress(min(1.0, done / max(1, total)))
                df_out = format_results_dataframe(results_all, multi_label=multi_label)
                df_final = pd.concat(
                    [
                        df_in.reset_index(drop=True),
                        df_out.drop(columns=["text"], errors="ignore"),
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
    "Notlar: 1) GPU kullanıyorsan doğru CUDA Torch paketini kur. "
    "2) Private HF repo için HUGGINGFACE_HUB_TOKEN ortam değişkenini ayarla. "
    "3) MODEL_REVISION vererek belirli bir commit'e sabitleyebilirsin."
)
