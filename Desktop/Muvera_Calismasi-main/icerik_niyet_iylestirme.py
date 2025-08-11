import os, re, json, subprocess, sys
import pandas as pd # type: ignore
from typing import Optional, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "prompts"))
from niyet_prompt import build_prompt  # type: ignore

# ======= Ayarlar =======
CSV_PATH = os.getenv("CSV_PATH", "html_icerik_niyet_uyumu.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "niyet_iyilestirme_sonuc.csv")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))  # Artırıldı
TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT", "120"))
MAX_IMPROVEMENT_ATTEMPTS = 3  # Pozitif skor için maksimum deneme

# ======= Skor modeli =======
from sentence_transformers import SentenceTransformer, util # type: ignore
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

def _similarity(a: str, b: str) -> float:
    a_emb = st_model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    b_emb = st_model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(a_emb, b_emb).item())

def _run_llm_with_improvement(kullanici_niyeti: str, mevcut_icerik: str, html_bolumu: str, eski_skor: float) -> Tuple[str, float]:
    """
    Pozitif skor elde edene kadar metni günceller.
    """
    best_candidate = mevcut_icerik
    best_score = eski_skor
    
    for improvement_attempt in range(MAX_IMPROVEMENT_ATTEMPTS):
        print(f"[INFO] İyileştirme denemesi {improvement_attempt + 1}/{MAX_IMPROVEMENT_ATTEMPTS}")
        
        # LLM ile içerik üret
        candidate = _run_llm_single_attempt(kullanici_niyeti, best_candidate, html_bolumu, best_score)
        
        # Yeni skoru hesapla
        new_score = _similarity(kullanici_niyeti, candidate)
        
        # Skor iyileşti mi kontrol et
        if new_score > best_score:
            print(f"[BAŞARILI] Skor iyileşti: {best_score:.6f} -> {new_score:.6f}")
            best_candidate = candidate
            best_score = new_score
            
            # Pozitif değişim elde edildi, döngüden çık
            if best_score > eski_skor:
                break
        else:
            print(f"[BAŞARISIZ] Skor iyileşmedi: {best_score:.6f} -> {new_score:.6f}")
            
            # Son denemede bile iyileşme olmadı, mevcut en iyi adayı kullan
            if improvement_attempt == MAX_IMPROVEMENT_ATTEMPTS - 1:
                print(f"[UYARI] Maksimum deneme sayısına ulaşıldı, en iyi aday kullanılıyor")
    
    return best_candidate, best_score

def _run_llm_single_attempt(kullanici_niyeti: str, mevcut_icerik: str, html_bolumu: str, eski_skor: float) -> str:
    """
    Tek bir LLM çağrısı yapar ve içeriği döndürür.
    """
    prompt = build_prompt(kullanici_niyeti, mevcut_icerik, html_bolumu, eski_skor)
    
    for attempt in range(MAX_RETRIES):
        completed = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=TIMEOUT_SEC,
        )
        
        if completed.returncode == 0:
            raw_output = completed.stdout.decode("utf-8", "ignore").strip()
            parsed = _extract_first_json(raw_output)
            if parsed and "Geliştirilmiş İçerik" in parsed:
                return parsed["Geliştirilmiş İçerik"]
        print(f"[UYARI] LLM cevabı alınamadı, deneme {attempt+1}/{MAX_RETRIES}")
    
    return mevcut_icerik  # Başarısızsa mevcut içeriği döndür

# ======= Veri =======
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="utf-8")
need = {"HTML Bölümü", "İçerik", "Kullanıcı Niyeti", "Benzerlik Skoru", "Uyum Durumu"}
miss = need - set(df.columns)
if miss:
    raise KeyError(f"Eksik kolonlar: {miss}")

df2 = df.copy()
uyum = df2["Uyum Durumu"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
score_raw = (df2["Benzerlik Skoru"].astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.extract(r"([-+]?\d*\.?\d+)", expand=False))
df2["Benzerlik Skoru"] = pd.to_numeric(score_raw, errors="coerce")

# Tüm uyumlu içerikleri ve 0.65-0.85 arası skorları işle
work = df2.loc[
    (uyum.eq("uyumlu")) | 
    (df2["Benzerlik Skoru"].between(0.65, 0.85, inclusive="both"))
].copy()

rows = []
for _, r in work.iterrows():
    intent = str(r["Kullanıcı Niyeti"]) if pd.notna(r["Kullanıcı Niyeti"]) else ""
    current = str(r["İçerik"]) if pd.notna(r["İçerik"]) else ""
    tag = (str(r["HTML Bölümü"]) if pd.notna(r["HTML Bölümü"]) else "p").lower()
    old = float(r["Benzerlik Skoru"]) if pd.notna(r["Benzerlik Skoru"]) else 0.0

    print(f"\n[İŞLENİYOR] Niyet: {intent}")
    print(f"[MEVCUT] Skor: {old:.6f}")
    
    # Pozitif skor elde edene kadar iyileştir
    cand, new_score = _run_llm_with_improvement(intent, current, tag, old)
    
    # Değişim yüzdesi
    change = ((new_score - old) / max(old, 1e-8)) * 100 if old > 0 else 0.0
    
    print(f"[SONUÇ] Eski: {old:.6f}, Yeni: {new_score:.6f}, Değişim: {change:.2f}%")
    
    rows.append({
        "Kullanıcı Niyeti": intent,
        "Mevcut İçerik": current,
        "Geliştirilmiş İçerik": cand,
        "HTML Bölümü": tag,
        "Eski Skor": round(old, 6),
        "Yeni Skor": round(float(new_score), 6),
        "Yüzde Değişim": round(change, 2),
    })

out = pd.DataFrame(rows, columns=[
    "Kullanıcı Niyeti", "Mevcut İçerik", "Geliştirilmiş İçerik",
    "HTML Bölümü", "Eski Skor", "Yeni Skor", "Yüzde Değişim"
])
print("\n" + "="*80)
print(out.to_string(index=False))
print("="*80)
out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\nTamamlandı. Çıktı: {OUTPUT_PATH}")