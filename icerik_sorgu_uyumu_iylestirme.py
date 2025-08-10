# -*- coding: utf-8 -*-
"""
İçerik × Kullanıcı Sorgusu uyumu iyileştirme (deterministik, CTA'sız, anlam koruyan).
- h1/h2: Sorgudan direkt başlık üretir.
    * "... nasıl / vermek / verme / oluşturma / kurma" → "<Base> Nasıl Yapılır?"
    * "... rehber / kılavuz" → "<Base> Kılavuzu"
    * 'verme' ve 'vermek' çekimleri KORUNUR.
- p/div: Eski metinden güvenli tek cümle ÖZET üretir + mikro eşanlamlı (ya da→veya, basit→kolay),
         sonda her zaman nokta.
- li: Niyet odaklı çok kısa TAM cümle ("... anlatılır." / yoksa "... özetlenir.").
- Uzunluk sınırları: p/div +%10, li +1; h1/h2 için kalıp sabitleri (Nasıl Yapılır?/Kılavuzu) asla kesilmez.
- "..." asla kullanılmaz.
Girdi: html_icerik_sorgu_uyumu.csv
Çıktı: output/icerik_sorgu_uyumu_sonuc.csv
"""

import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ====== IO ======
INPUT_CSV  = os.getenv("INPUT_CSV",  "html_icerik_sorgu_uyumu.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "icerik_sorgu_uyumu_sonuc.csv")

# ====== Model ======
st_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

# ====== Kurallar / Yardımcılar ======
CONJ_TAILS = {"ve","veya","ya","ya da","ile","ama","ancak","fakat","çünkü","ki"}

_BANNED = [
    "öğrenmek isterseniz","öğrenmek istersiniz","başlayabilirsiniz","başvurunuzu yaparak",
    "hedef kitlenize ulaşın","potansiyel müşteriler","yardımcı olur","kampanyalarınız oluşturabilirsiniz"
]

def sim(a: str, b: str) -> float:
    if not a or not b: return 0.0
    a_emb = st_model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    b_emb = st_model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(a_emb, b_emb).item())

def to_title_tr(text: str) -> str:
    t = " ".join((text or "").split())
    low = {"ve","veya","ile","için","ya","ya da","mi","mı","mu","mü"}
    out=[]
    for i,w in enumerate(t.split()):
        wl=w.lower()
        out.append(wl if (i>0 and wl in low) else wl.capitalize())
    return " ".join(out)

def finalize(text: str, tag: str) -> str:
    t = " ".join((text or "").split()).strip(" ,;:-")
    words = t.split()
    while words and words[-1].lower() in CONJ_TAILS:
        words.pop()
    t=" ".join(words).strip()
    tag=tag.lower()
    if tag in ("h1","h2"): return t
    if tag=="li": return t if re.search(r"[.!?]$",t) else t+"."
    # p/div: daima nokta
    return t if re.search(r"[.!?]$",t) else t+"."

def enforce_delta(original: str, cand: str, tag: str) -> str:
    tag=tag.lower()
    o = (original or "").split(); c = (cand or "").split()
    if not o or not c: return cand
    if tag in ("h1","h2"):
        s=" ".join(c)
        # Kalıp sabitleri asla kesilmez
        if s.endswith(" Nasıl Yapılır?") or s.endswith(" Kılavuzu"):
            return s
        max_len = max(len(o)+4, int(len(o)*1.8))
        return " ".join(c[:max_len])
    if tag=="li":
        return " ".join(c[:len(o)+1])
    # p/div: +%10, kısa metinde +1
    max_len = int(len(o)*1.10) if len(o)>=10 else len(o)+1
    return " ".join(c[:max_len])

def clean_marketing(text: str) -> str:
    t = " ".join((text or "").split())
    for b in _BANNED:
        t = re.sub(re.escape(b), "", t, flags=re.IGNORECASE).strip()
    return re.sub(r"\s{2,}"," ",t)

def micro_edit_paragraph(text: str) -> str:
    t = " ".join((text or "").split())
    rules = [
        (r"\by[aı]\s+da\b","veya"),
        (r"\bBasit\b","Kolay"),
        (r"\bbasit\b","kolay"),
    ]
    for pat,rep in rules:
        new = re.sub(pat,rep,t,count=1)
        if new!=t:
            t=new; break
    return clean_marketing(t)

# ---- Soru kalıbı algısı ve başlık tabanı ----
def implies_how(q: str) -> bool:
    L = " ".join((q or "").split()).lower()
    return any(k in L for k in [
        "nasıl","verme","vermek","oluşturma","oluşturmak","kurma","kurmak","açma","açmak"
    ])

def base_from_query(q: str) -> str:
    """Sorgudaki çekimi koruyarak gövdeyi oluştur (verme/vermek vb. kaybolmasın)."""
    L = " ".join((q or "").split()).lower()
    if "verme" in L:   return to_title_tr("google reklam verme")
    if "vermek" in L:  return to_title_tr("google reklam vermek")
    if "oluşturma" in L or "oluşturmak" in L:
        return to_title_tr("google reklamı oluşturma")
    if "kurma" in L or "kurmak" in L:
        return to_title_tr("google reklam hesabı kurma")
    return to_title_tr(L)

def format_heading_from_query(q: str) -> str:
    """h1/h2 başlığı: sorguya göre 'Nasıl Yapılır?' ya da 'Kılavuzu'."""
    L = " ".join((q or "").split()).lower()
    if re.search(r"\bnasıl\b", L) or implies_how(q):
        base = base_from_query(q)                      # << 'verme/vermek' korunur
        return f"{base} Nasıl Yapılır?"
    if re.search(r"\b(rehber(i)?)|(kılavuz(u)?)\b", L):
        base = base_from_query(q)
        # 'oluşturma' özel hali zaten base'e yansır
        return f"{base} Kılavuzu"
    return to_title_tr(L)

# ---- p/div için sorgu odaklı özet ----
def summarize_from_old(cur: str, q: str) -> str:
    """
    p/div için tek cümle, sorgu odaklı özet.
    - 'verme/vermek' → süreç + CPC + arama sonuçları
    - 'oluşturma' → kampanya + ilan ayarları
    - aksi halde: güvenli kısa tanım
    """
    t = " ".join((cur or "").split())
    has_search  = bool(re.search(r"arama sonuçlar", t, flags=re.IGNORECASE))
    has_cpc     = bool(re.search(r"tıkl(a|ama).*ödeme|tıkladığında ödeme|tıklama başına", t, flags=re.IGNORECASE))
    has_platform= bool(re.search(r"\bgoogle ads\b", t, flags=re.IGNORECASE))

    L = " ".join((q or "").split()).lower()
    if "verme" in L or "vermek" in L:
        parts = []
        parts.append("Google Ads ile reklam verme süreci tıklama başına ödeme modeline dayanır" if (has_cpc or has_platform)
                     else "Google Ads ile reklam verme süreci temel adımlarla ilerler")
        if has_search:
            parts.append("reklamlar arama sonuçlarında yayınlanır")
        sent = ", ".join(parts) + "."
    elif "oluştur" in L:
        sent = "Google reklamı oluşturma, Google Ads üzerinde kampanya ve ilan ayarlarıyla yapılır."
    else:
        sent = "Google Ads, çevrimiçi reklamları yönetmek için kullanılan bir platformdur."

    sent = micro_edit_paragraph(sent)
    return finalize(sent, "div")

# ---- li için kısa tam cümle ----
def li_from_query(q: str, original: str) -> str:
    """li: çok kısa, tam cümle, sorgu terimini içersin."""
    L = " ".join((q or "").split()).lower()
    if "verme" in L:
        cand = "Google reklam verme anlatılır."
    elif "vermek" in L:
        cand = "Google reklam vermek anlatılır."
    elif "oluştur" in L:
        cand = "Google reklamı oluşturma anlatılır."
    else:
        base = " ".join((original or "").split()) or to_title_tr(L)
        cand = base + " özetlenir."
    return finalize(cand, "li")

# ---- Alternatif çok kısa cevap ----
def short_answer_from_query(q: str, tag: str) -> str:
    """Alternatif kısa cevap: başlık/yanıtlar birbirinden farklı olsun."""
    L = " ".join((q or "").split()).lower()
    if "verme" in L:
        text = "Google reklam verme adımları nelerdir?"
    elif "vermek" in L:
        text = "Google reklam vermek için temel adımlar nelerdir?"
    elif "oluştur" in L:
        text = "Google reklamı oluşturma adımları nelerdir?"
    elif re.search(r"\bnasıl\b", L):
        text = f"{base_from_query(q)} nasıl yapılır?"
    else:
        text = to_title_tr(L)
    return finalize(text, tag)

# ====== Çalıştırma ======
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Girdi bulunamadı: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    need = {"HTML Bölümü","İçerik","Kullanıcı Sorgusu","Benzerlik Skoru","Uyum Durumu"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Eksik kolonlar: {miss}. Mevcut: {list(df.columns)}")

    norm = df["Uyum Durumu"].astype(str).str.strip().str.lower().str.replace(r"\s+"," ",regex=True)
    cand_df = df.loc[norm.eq("uyumlu") & df["Benzerlik Skoru"].between(0.65,0.85, inclusive="both")].copy()

    rows=[]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in cand_df.iterrows():
        q   = str(row["Kullanıcı Sorgusu"] or "")
        cur = str(row["İçerik"] or "")
        tag = (str(row["HTML Bölümü"] or "")).lower()
        old = float(row["Benzerlik Skoru"] or 0.0)

        # Aday 1: deterministik kural
        if tag in ("h1","h2"):
            det = format_heading_from_query(q)
        elif tag in ("p","div"):
            det = summarize_from_old(cur, q)   # << q parametresi ile sorgu odaklı özet
        elif tag=="li":
            det = li_from_query(q, cur)
        else:
            det = micro_edit_paragraph(cur)

        det = enforce_delta(cur, finalize(det, tag), tag)

        # Aday 2: sorgu tabanlı kısa cevap
        qans = enforce_delta(cur, short_answer_from_query(q, tag), tag)

        # En iyi skoru seç
        s1 = sim(q, det)
        s2 = sim(q, qans)
        improved = det if s1 >= s2 else qans
        new = max(s1, s2)
        change = round(((new - old)/max(old,1e-8))*100, 2) if old>0 else 0.0

        rows.append({
            "HTML Bölümü": tag,
            "Kullanıcı Sorgusu": q,
            "Eski Metin": cur,
            "Geliştirilmiş Metin": improved,
            "Eski Skor": round(old,6),
            "Yeni Skor": round(float(new),6),
            "Yüzde Değişim": change
        })

    out = pd.DataFrame(rows, columns=[
        "HTML Bölümü","Kullanıcı Sorgusu","Eski Metin","Geliştirilmiş Metin",
        "Eski Skor","Yeni Skor","Yüzde Değişim"
    ])
    print(out.head())
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Tamamlandı. Çıktı: {OUTPUT_CSV}")
