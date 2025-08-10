"""
0.65–0.85 arası benzerlik skoruna sahip UYUMLU satırları küçük dokunuşlarla iyileştirir.
- h1/h2: Niyetten DOĞRUDAN başlık üretir (örn. "google reklam verme nasıl" → "Google Reklam Verme Nasıl Yapılır?").
- p/div: Anlamı bozmadan tek kelimelik eşanlamlı düzeltme (ya da→veya, basit→kolay). CTA/pazarlama ifadesi eklemez.
- li: Niyete göre kısa, TAM cümle (“… anlatılır.” / “… özetlenir.”), en fazla +1 kelime oynar.
- p/div +%10; li +1 sınırı; h1/h2 için “Nasıl Yapılır?”/“Kılavuzu” asla kesilmez.
- Rollback yok; her zaman değiştirir.
"""

import os, re, json, subprocess, sys
import pandas as pd
from typing import Optional, Dict
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

# ======= Skor modeli =======
from sentence_transformers import SentenceTransformer, util
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
st_model = SentenceTransformer(ST_MODEL_NAME)

CONJ_TAILS = {"ve","veya","ya","ya da","ile","ama","ancak","fakat","çünkü","ki"}
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_BANNED_PHRASES = [
    "öğrenmek isterseniz","öğrenmek istersiniz","başlayabilirsiniz","başvurunuzu yaparak",
    "hedef kitlenize ulaşın","potansiyel müşteriler","yardımcı olur","kampanyalarınız oluşturabilirsiniz"
]

def _extract_first_json(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def _similarity(a: str, b: str) -> float:
    a_emb = st_model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    b_emb = st_model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(a_emb, b_emb).item())

def _to_title_tr(text: str) -> str:
    t = " ".join((text or "").split())
    low = {"ve","veya","ile","için","ya","ya da","mi","mı","mu","mü"}
    words = t.split()
    out = []
    for i,w in enumerate(words):
        wl = w.lower()
        out.append(wl if (i>0 and wl in low) else wl.capitalize())
    return " ".join(out)

def _finalize_sentence(text: str, tag: str) -> str:
    t = " ".join((text or "").split()).strip(" ,;:-")
    words = t.split()
    while words and words[-1].lower() in CONJ_TAILS: words.pop()
    t = " ".join(words).strip()
    tag_l = (tag or "").lower()
    if tag_l in ("h1","h2"): return t
    if tag_l=="li": return t if re.search(r"[.!?]$", t) else t+"."
    return t if re.search(r"[.!?]$", t) else t+"."

def _enforce_word_delta(original: str, candidate: str, tag: str) -> str:
    tag_l = (tag or "").lower()
    o_words = (original or "").split(); c_words = (candidate or "").split()
    if not o_words or not c_words: return candidate

    if tag_l in ("h1","h2"):
        # “Nasıl Yapılır?” veya “Kılavuzu” varsa ASLA kesme
        cand_str = " ".join(c_words)
        if (" Nasıl Yapılır?" in cand_str) or cand_str.endswith(" Kılavuzu") or cand_str.endswith(" Kılavuzu?"):
            return cand_str
        # aksi halde geniş tavan: +4 kelime ya da *1.8
        max_len = max(len(o_words)+4, int(len(o_words)*1.8))
        if len(c_words)>max_len: c_words=c_words[:max_len]
        return " ".join(c_words)

    if tag_l=="li":
        max_len = len(o_words)+1
        if len(c_words)>max_len: c_words=c_words[:max_len]
        return " ".join(c_words)

    # p/div
    max_len = int(len(o_words)*1.10) if len(o_words)>=10 else len(o_words)+1
    if len(c_words)>max_len: c_words=c_words[:max_len]
    return " ".join(c_words)

def _sanitize_no_marketing(text: str) -> str:
    t = " ".join((text or "").split())
    for b in _BANNED_PHRASES:
        t = re.sub(re.escape(b), "", t, flags=re.IGNORECASE).strip()
    return re.sub(r"\s{2,}"," ",t)

def _micro_edit_paragraph_strict(original: str) -> str:
    # Tek bir küçük eşanlamlı dokunuş (anlamı koru)
    t = " ".join((original or "").split())
    rules = [
        (r"\by[aı]\s+da\b","veya"),
        (r"\bBasit\b","Kolay"),
        (r"\bbasit\b","kolay"),
    ]
    for pat,rep in rules:
        new = re.sub(pat,rep,t,count=1)
        if new!=t:
            t=new; break
    return _sanitize_no_marketing(t)

def _format_heading_from_intent(intent: str) -> str:
    L = " ".join((intent or "").split()).lower()
    if re.search(r"\bnasıl\b", L):
        prefix = re.sub(r"\bnasıl\b","",L).strip()
        prefix = re.sub(r"\bverme\b","verme",prefix)
        prefix = re.sub(r"\boluşturma\b","oluşturma",prefix)
        return f"{_to_title_tr(prefix)} Nasıl Yapılır?"
    if re.search(r"\b(rehber(i)?)|(kılavuz(u)?)\b", L):
        base = re.sub(r"\b(rehber(i)?)|(kılavuz(u)?)\b","",L).strip()
        base_t = _to_title_tr(base)
        if re.search(r"\boluştur(ma)?\b", L) and "Oluşturma" not in base_t:
            base_t = f"{base_t} Oluşturma"
        return f"{base_t} Kılavuzu"
    return _to_title_tr(L)

def _tweak_li_minimal(original: str, intent: str) -> str:
    base = " ".join((original or "").split())
    L = " ".join((intent or "").split()).lower()
    if "oluştur" in L:
        cand = f"{_to_title_tr('Google Reklamı oluşturma')} anlatılır."
    elif "verme" in L:
        cand = f"{_to_title_tr('Google reklam verme')} anlatılır."
    else:
        cand = base.rstrip(".")+" özetlenir."
    cand = _finalize_sentence(cand, "li")
    return _enforce_word_delta(base, cand, "li")

def _run_llm(kullanici_niyeti: str, mevcut_icerik: str, html_bolumu: str, eski_skor: float) -> str:
    # Halen kullanılmıyor; ileride gerekirse.
    prompt = build_prompt(kullanici_niyeti, mevcut_icerik, html_bolumu, eski_skor)
    completed = subprocess.run(
        ["ollama","run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=TIMEOUT_SEC,
    )
    if completed.returncode!=0:
        raise RuntimeError(completed.stderr.decode("utf-8","ignore"))
    return completed.stdout.decode("utf-8","ignore").strip()

# ======= Veri =======
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8")
need = {"HTML Bölümü","İçerik","Kullanıcı Niyeti","Benzerlik Skoru","Uyum Durumu"}
miss = need - set(df.columns)
if miss: raise KeyError(f"Eksik kolonlar: {miss}")

df2 = df.copy()
uyum = df2["Uyum Durumu"].astype(str).str.strip().str.lower().str.replace(r"\s+"," ",regex=True)
score_raw = (df2["Benzerlik Skoru"].astype(str)
    .str.replace("%","",regex=False).str.replace(",",".",regex=False)
    .str.extract(r"([-+]?\d*\.?\d+)", expand=False))
df2["Benzerlik Skoru"] = pd.to_numeric(score_raw, errors="coerce")

work = df2.loc[uyum.eq("uyumlu") & df2["Benzerlik Skoru"].between(0.65,0.85, inclusive="both")].copy()

rows=[]
for _, r in work.iterrows():
    intent = str(r["Kullanıcı Niyeti"]) if pd.notna(r["Kullanıcı Niyeti"]) else ""
    current = str(r["İçerik"]) if pd.notna(r["İçerik"]) else ""
    tag = (str(r["HTML Bölümü"]) if pd.notna(r["HTML Bölümü"]) else "p").lower()
    old = float(r["Benzerlik Skoru"]) if pd.notna(r["Benzerlik Skoru"]) else 0.0

    if tag in ("h1","h2"):
        cand = _format_heading_from_intent(intent)
        cand = _finalize_sentence(_enforce_word_delta(current, cand, tag), tag)
    elif tag in ("p","div"):
        cand = _micro_edit_paragraph_strict(current)
        cand = _finalize_sentence(_enforce_word_delta(current, cand, tag), tag)
    elif tag=="li":
        cand = _tweak_li_minimal(current, intent)
    else:
        tmp = _micro_edit_paragraph_strict(current)
        cand = _finalize_sentence(_enforce_word_delta(current, tmp, tag), tag)

    new = _similarity(intent, cand)
    change = ((new-old)/max(old,1e-8))*100 if old>0 else 0.0

    rows.append({
        "Kullanıcı Niyeti": intent,
        "Mevcut İçerik": current,
        "Geliştirilmiş İçerik": cand,
        "HTML Bölümü": tag,
        "Eski Skor": round(old,6),
        "Yeni Skor": round(float(new),6),
        "Yüzde Değişim": round(change,2),
    })

out = pd.DataFrame(rows, columns=[
    "Kullanıcı Niyeti","Mevcut İçerik","Geliştirilmiş İçerik",
    "HTML Bölümü","Eski Skor","Yeni Skor","Yüzde Değişim"
])
print(out.head())
out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Tamamlandı. Çıktı: {OUTPUT_PATH}")
