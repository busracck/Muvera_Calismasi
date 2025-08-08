"""
0.65–0.85 arası benzerlik skoruna sahip UYUMLU satırları iyileştirir.
- LLM’den gelen öneriyi uzunluk/etiket/dilbilgisi kurallarına göre normalize eder.
- Intent terimi kapsamasını ve yeni skorunu hesaplar.
- İyileşme yoksa rollback yapar (mevcut içeriği korur).
- Çıktı kolonları: Kullanıcı Niyeti, Mevcut İçerik, Geliştirilmiş İçerik, HTML Bölümü, Eski Skor, Yeni Skor, Yüzde Değişim
"""

import os
import re
import json
import subprocess
import sys
import pandas as pd
from typing import Optional, Dict

# prompts klasörünü modül yoluna ekle
sys.path.append(os.path.join(os.path.dirname(__file__), "prompts"))
from niyet_prompt import build_prompt  # type: ignore

# ======= Ayarlar =======
CSV_PATH = os.getenv("CSV_PATH", "html_icerik_niyet_uyumu.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "niyet_iyilestirme_sonuc.csv")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ======= Embedding / Skor modeli =======
from sentence_transformers import SentenceTransformer, util
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
st_model = SentenceTransformer(ST_MODEL_NAME)

# ======= Yardımcılar =======

def _extract_first_json(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _terms(s: str):
    return set(w.lower() for w in re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", s or ""))


def _similarity(a: str, b: str) -> float:
    a_emb = st_model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    b_emb = st_model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(a_emb, b_emb).item())


CONJ_TAILS = {"ve", "veya", "ya", "ya da", "ile", "ama", "ancak", "fakat", "çünkü", "ki"}

# Cümleye göre kes – kelimeye göre değil
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _truncate_to_sentences(text: str, tag: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return t
    parts = _SENT_SPLIT.split(t)
    tag_l = (tag or "").lower()
    if tag_l in ("h1", "h2", "li"):
        return parts[0].strip()
    return " ".join(parts[:2]).strip()


def _finalize_sentence(text: str, tag: str) -> str:
    t = " ".join((text or "").split()).strip(" ,;:-")
    words = t.split()
    while words and words[-1].lower() in CONJ_TAILS:
        words.pop()
    t = " ".join(words).strip()
    tag_l = (tag or "").lower()
    if tag_l in ("h1", "h2"):
        return t  # başlıklara nokta ekleme
    if tag_l == "li":
        return t if re.search(r"[.!?]$", t) else t + "."
    return t if re.search(r"[.!?]$", t) else t + "."

# Türkçe "nasıl" vb. düzeltmeler ve soru izni
_DEF_VERB_FIXES = [
    (r"(?i)\b(nasıl\s+[^\s]+)\s+verme\b", r"\1 verilir"),
    (r"(?i)\b(nasıl\s+[^\s]+)\s+yapma\b", r"\1 yapılır"),
    (r"(?i)\b(nasıl\s+[^\s]+)\s+alma\b", r"\1 alınır"),
    (r"(?i)\b(nasıl\s+[^\s]+)\s+kapama\b", r"\1 kapatılır"),
    (r"(?i)\b(nasıl\s+[^\s]+)\s+açma\b", r"\1 açılır"),
    (r"(?i)\b([^\s]+)\s+nasıl\s+verme\b", r"\1 nasıl verilir"),
    (r"(?i)\b([^\s]+)\s+nasıl\s+yapma\b", r"\1 nasıl yapılır"),
    (r"(?i)\b([^\s]+)\s+nasıl\s+alma\b", r"\1 nasıl alınır"),
    (r"(?i)\b([^\s]+)\s+nasıl\s+kapama\b", r"\1 nasıl kapatılır"),
    (r"(?i)\b([^\s]+)\s+nasıl\s+açma\b", r"\1 nasıl açılır"),
]

def _allow_question(intent: str) -> bool:
    L = intent.lower()
    return any(k in L for k in ["nasıl", "nedir", " mi ", " mı ", " mu ", " mü "])


def _fix_how_clause(text: str, tag: str, allow_q: bool) -> str:
    t = " ".join((text or "").split())
    for pat, rep in _DEF_VERB_FIXES:
        t = re.sub(pat, rep, t)

    tag_l = (tag or "").lower()
    if "nasıl" in t.lower():
        if tag_l in ("h1", "h2"):
            if not allow_q:
                t = t.rstrip("?.")
        else:
            if allow_q:
                t = t.rstrip(".") + "?"
            else:
                t = t.rstrip("?") + "."
    return t

# LLM çağrısı

def _run_llm(kullanici_niyeti: str, mevcut_icerik: str, html_bolumu: str, eski_skor: float) -> str:
    prompt = build_prompt(kullanici_niyeti, mevcut_icerik, html_bolumu, eski_skor)
    completed = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=TIMEOUT_SEC,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Ollama hata döndürdü: {completed.stderr.decode('utf-8', errors='ignore')}"
        )
    return completed.stdout.decode("utf-8", errors="ignore").strip()

# ======= Veri yükleme =======
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8")

# Kolon doğrulama
needed_cols = {"HTML Bölümü", "İçerik", "Kullanıcı Niyeti", "Benzerlik Skoru", "Uyum Durumu"}
missing = needed_cols - set(df.columns)
if missing:
    raise KeyError(f"Beklenen kolonlar eksik: {missing}. Mevcut kolonlar: {list(df.columns)}")

# Dayanıklı normalizasyon ve filtre
_df = df.copy()
uyum_norm = (
    _df["Uyum Durumu"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
)
score_raw = (
    _df["Benzerlik Skoru"].astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
)
_df["Benzerlik Skoru"] = pd.to_numeric(score_raw, errors="coerce")

mask = uyum_norm.eq("uyumlu") & _df["Benzerlik Skoru"].between(0.65, 0.85, inclusive="both")
work = _df.loc[mask].copy()

if work.empty:
    print("⚠️ Filtre sonrası satır yok.")
    print("Uyum Durumu örnekleri:", uyum_norm.value_counts().head(10).to_dict())
    print("Skor min/max:", _df["Benzerlik Skoru"].min(), _df["Benzerlik Skoru"].max())
    print("NaN skor sayısı:", _df["Benzerlik Skoru"].isna().sum())

rows = []
for _, r in work.iterrows():
    intent = str(r["Kullanıcı Niyeti"]) if pd.notna(r["Kullanıcı Niyeti"]) else ""
    current = str(r["İçerik"]) if pd.notna(r["İçerik"]) else ""
    tag = str(r["HTML Bölümü"]) if pd.notna(r["HTML Bölümü"]) else "p"
    old_score = float(r["Benzerlik Skoru"]) if pd.notna(r["Benzerlik Skoru"]) else 0.0

    best_text = None
    best_score = -1.0
    best_cov = False

    intent_terms = _terms(intent)
    allow_q = _allow_question(intent)

    for _try in range(MAX_RETRIES):
        raw = _run_llm(intent, current, tag, old_score)
        data = _extract_first_json(raw) or {}
        cand = data.get("gelistirilmis_icerik") or current

        # Normalize et: önce cümleye göre kes, sonra soru kipini düzelt, en sonda sonlandır
        cand = _truncate_to_sentences(cand, tag)
        cand = _fix_how_clause(cand, tag, allow_q)
        cand = _finalize_sentence(cand, tag)

        # Kapsama ve skor
        out_terms = _terms(cand)
        coverage_ok = len(intent_terms & out_terms) >= max(1, len(intent_terms) // 2)
        new_score = _similarity(intent, cand)

        if new_score > best_score:
            best_score, best_text, best_cov = new_score, cand, coverage_ok

        if (new_score >= old_score) and coverage_ok:
            break

    if (best_score < old_score) or (not best_cov):
        improved = current
        new_score = old_score
        change_str = "değişmedi (rollback)"
    else:
        improved = best_text
        new_score = best_score
        if abs(new_score - old_score) < 1e-8:
            change_str = "değişmedi"
        else:
            change_pct = ((new_score - old_score) / max(old_score, 1e-8)) * 100
            change_str = round(change_pct, 2)

    rows.append({
        "Kullanıcı Niyeti": intent,
        "Mevcut İçerik": current,
        "Geliştirilmiş İçerik": improved,
        "HTML Bölümü": tag,
        "Eski Skor": round(old_score, 6),
        "Yeni Skor": round(float(new_score), 6),
        "Yüzde Değişim": change_str,
    })

out_df = pd.DataFrame(rows, columns=[
    "Kullanıcı Niyeti",
    "Mevcut İçerik",
    "Geliştirilmiş İçerik",
    "HTML Bölümü",
    "Eski Skor",
    "Yeni Skor",
    "Yüzde Değişim",
])

print(out_df.head())

out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Tamamlandı. Çıktı: {OUTPUT_PATH}")