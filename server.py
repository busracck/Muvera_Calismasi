# server.py (veya FastAPI dosyan)
from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import pandas as pd
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def _num(x):
    if x is None: return None
    s = str(x).strip()
    s = s.replace("%","").replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else None

@app.get("/api/results/niyet")
def results_niyet():
    path = "output/niyet_iyilestirme_sonuc.csv"
    df = pd.read_csv(path)

    need = {"HTML Bölümü","Kullanıcı Niyeti","Mevcut İçerik",
            "Geliştirilmiş İçerik","Eski Skor","Yeni Skor","Yüzde Değişim"}
    missing = need - set(df.columns)
    if missing:
        return {"rows": [], "error": f"CSV kolonları eksik: {missing}"}

    rows = []
    for _, r in df.iterrows():
        row = {
            "html":        str(r["HTML Bölümü"]),
            "context":     str(r["Kullanıcı Niyeti"]),      # İÇERİK × NİYET görünümü
            "oldText":     str(r["Mevcut İçerik"]),
            "newText":     str(r["Geliştirilmiş İçerik"]),
            "oldScore":    _num(r["Eski Skor"]),
            "newScore":    _num(r["Yeni Skor"]),
            "delta":       _num(r["Yüzde Değişim"]),        # yüzde işareti/virgül temizlendi
        }
        rows.append(row)

    # İstersen buradaki eşik sunucuda da uygulanabilir (frontend zaten >0.001 filtreliyor)
    rows = [r for r in rows if r["delta"] is not None and r["delta"] > 0.001]
    return {"rows": rows}
