# -*- coding: utf-8 -*-
from ollama import Client
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import json
import logging
from prompts.sorgu_prompt import get_sorgu_iyilestirme_prompt

# ---- Config ----
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3:instruct"   # gerekirse: gemma3:4b
TEMPERATURE = 0.2
MAX_RETRIES = 4
MIN_DELTA = 0.01  # min +%1 iyileşme

INPUT_CSV  = "html_icerik_sorgu_uyumu.csv"
OUTPUT_CSV = "daha_uyumlu_sorgu_oneriler_yuzde_degisim.csv"

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("sorgu_iyilestirme")

# ---- Models ----
ollama_client = Client(host=OLLAMA_HOST)
st_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

# ---- Helpers ----
def clamp_words(text: str, max_words: int) -> str:
    return " ".join((text or "").split()[:max_words])

def finalize_sentence(text: str, html_tag: str) -> str:
    t = " ".join((text or "").split()).strip(" ,;:-")
    tag = (html_tag or "").lower()
    if tag in ("h1", "h2"):  # başlıkta sonda nokta yok
        return t
    # li/p/div tam cümle bitişi
    if not re.search(r"[.!?]$", t):
        if re.search(r"(?i)\bnasıl\b", t):
            return t + "?"
        return t + "."
    return t

def fix_how_general(text: str) -> str:
    """Genel 'nasıl ... yapma/verme/açma/kapama' → '... yapılır/verilir/açılır/kapatılır' dönüşümü + soru işareti."""
    t = " ".join((text or "").split())
    # “nasıl X yapma” vb.
    t = re.sub(r'(?i)\b(nasıl\s+[^\s]+)\s+yapma\b',  r'\1 yapılır', t)
    t = re.sub(r'(?i)\b(nasıl\s+[^\s]+)\s+verme\b',  r'\1 verilir', t)
    t = re.sub(r'(?i)\b(nasıl\s+[^\s]+)\s+açma\b',   r'\1 açılır',  t)
    t = re.sub(r'(?i)\b(nasıl\s+[^\s]+)\s+kapama\b', r'\1 kapatılır', t)
    # “X nasıl yapma” vb.
    t = re.sub(r'(?i)\b([^\s]+)\s+nasıl\s+yapma\b',  r'\1 nasıl yapılır', t)
    t = re.sub(r'(?i)\b([^\s]+)\s+nasıl\s+verme\b',  r'\1 nasıl verilir', t)
    t = re.sub(r'(?i)\b([^\s]+)\s+nasıl\s+açma\b',   r'\1 nasıl açılır',  t)
    t = re.sub(r'(?i)\b([^\s]+)\s+nasıl\s+kapama\b', r'\1 nasıl kapatılır', t)
    # 'nasıl' var ve nokta ile bitiyorsa soru işaretine çevir
    if re.search(r'(?i)\bnasıl\b', t) and t.endswith('.'):
        t = t[:-1] + '?'
    return t

def hesapla_benzerlik(sorgu: str, metin: str) -> float:
    if not sorgu or not metin:
        return 0.0
    s = st_model.encode(sorgu, convert_to_tensor=True, normalize_embeddings=True)
    m = st_model.encode(metin, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(s, m).item())

def safe_percent_change(old: float, new: float) -> str:
    try:
        old = float(old or 0.0); new = float(new or 0.0)
        if old <= 0: return "yeni skor hesaplandı"
        pct = round(((new - old) / old) * 100, 2)
        return f"%{pct} artmış" if pct >= 0 else f"%{abs(pct)} azalmış"
    except Exception:
        return "değişmedi"

def chat_llm(sorgu: str, mevcut_metin: str, html_bolumu: str) -> str:
    prompt = get_sorgu_iyilestirme_prompt()
    messages = prompt.format_messages(sorgu=sorgu, mevcut_metin=mevcut_metin, html_bolumu=html_bolumu)
    msgs = [m.dict() for m in messages]  # system + user
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            resp = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=msgs,
                options={'temperature': TEMPERATURE}
            )
            return (resp.get("message") or {}).get("content", "")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LLM hata: {last_err}")

def mini_patch(sorgu: str, text: str, html_bolumu: str) -> str:
    cand = (text or "").strip()
    if not cand:
        cand = sorgu
    # Başlık/LI kısa tut
    tag = (html_bolumu or "").lower()
    cand = clamp_words(cand, 18 if tag in ("h1","h2","li") else 40)
    cand = fix_how_general(cand)
    cand = finalize_sentence(cand, tag)
    return cand

def forced_patch(sorgu: str, html_bolumu: str) -> str:
    tag = (html_bolumu or "").lower()
    forced = f"{sorgu}"
    forced = clamp_words(forced, 18 if tag in ("h1","h2","li") else 40)
    forced = fix_how_general(forced)
    forced = finalize_sentence(forced, tag)
    return forced

# ---- Main ----
df = pd.read_csv(INPUT_CSV)

# Beklenen kolonlar:
# "HTML Bölümü", "İçerik", "Kullanıcı Sorgusu", "Benzerlik Skoru", "Uyum Durumu"
missing = {"HTML Bölümü","İçerik","Kullanıcı Sorgusu","Benzerlik Skoru","Uyum Durumu"} - set(df.columns)
if missing:
    raise KeyError(f"Kolon eksik: {missing}. Mevcut: {list(df.columns)}")

hedefler = df[
    (df["Uyum Durumu"].astype(str).str.lower() == "uyumlu") &
    (df["Benzerlik Skoru"] > 0.65) &
    (df["Benzerlik Skoru"] < 0.85)
].copy()

log.info("Toplam geliştirilecek içerik sayısı: %d", len(hedefler))
sonuclar = []

for _, row in hedefler.iterrows():
    sorgu = str(row["Kullanıcı Sorgusu"] or "")
    mevcut = str(row["İçerik"] or "")
    html   = str(row["HTML Bölümü"] or "")
    eski_skor = float(row["Benzerlik Skoru"] or 0.0)

    try:
        # 0) LLM üretim
        raw = chat_llm(sorgu, mevcut, html)

        # 1) Patch ve skor
        cand = mini_patch(sorgu, raw, html)
        yeni_skor = hesapla_benzerlik(sorgu, cand)

        # 2) Zorunlu iyileştirme (eşik altı ise fallback)
        if yeni_skor < (eski_skor + MIN_DELTA):
            forced = mini_patch(sorgu, forced_patch(sorgu, html), html)
            f_skor = hesapla_benzerlik(sorgu, forced)
            if f_skor > yeni_skor:
                cand, yeni_skor = forced, f_skor

        sonuclar.append({
            "HTML Bölümü": html,
            "Kullanıcı Sorgusu": sorgu,
            "Eski Metin": mevcut,
            "Geliştirilmiş Metin": cand,
            "Eski Skor": round(eski_skor, 6),
            "Yeni Skor": round(yeni_skor, 6),
            "Yüzde Değişim": safe_percent_change(eski_skor, yeni_skor)
        })

    except Exception as e:
        log.exception("Hata: '%s' → %s", sorgu, e)

pd.DataFrame(sonuclar).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Geliştirme tamamlandı. Çıktı: {OUTPUT_CSV}")
