# server.py
from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import os, pandas as pd
from typing import Dict, List, Literal

OUT_DIR = os.getenv("OUTPUT_DIR", "output")
NIYET_CSV = os.path.join(OUT_DIR, "niyet_iyilestirme_sonuc.csv")
SORGU_CSV = os.path.join(OUT_DIR, "icerik_sorgu_uyumu_sonuc.csv")

app = FastAPI(title="Muvera API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def _to_num(v):
    try: return float(str(v).replace(",", "."))
    except: return None

def _normalize(df: pd.DataFrame, kind: Literal["niyet","sorgu"]) -> List[Dict]:
    rows=[]
    if kind=="niyet":
        need=["HTML Bölümü","Kullanıcı Niyeti","Mevcut İçerik","Geliştirilmiş İçerik","Eski Skor","Yeni Skor","Yüzde Değişim"]
        if any(c not in df.columns for c in need):
            raise HTTPException(400, detail=f"Eksik kolon: {need}")
        for _,r in df.iterrows():
            rows.append({
                "html": r["HTML Bölümü"] or "",
                "context": r["Kullanıcı Niyeti"] or "",
                "oldText": r["Mevcut İçerik"] or "",
                "newText": r["Geliştirilmiş İçerik"] or "",
                "oldScore": _to_num(r["Eski Skor"]),
                "newScore": _to_num(r["Yeni Skor"]),
                "delta": _to_num(r["Yüzde Değişim"]),
            })
    else:
        need=["HTML Bölümü","Kullanıcı Sorgusu","Eski Metin","Geliştirilmiş Metin","Eski Skor","Yeni Skor","Yüzde Değişim"]
        if any(c not in df.columns for c in need):
            raise HTTPException(400, detail=f"Eksik kolon: {need}")
        for _,r in df.iterrows():
            rows.append({
                "html": r["HTML Bölümü"] or "",
                "context": r["Kullanıcı Sorgusu"] or "",
                "oldText": r["Eski Metin"] or "",
                "newText": r["Geliştirilmiş Metin"] or "",
                "oldScore": _to_num(r["Eski Skor"]),
                "newScore": _to_num(r["Yeni Skor"]),
                "delta": _to_num(r["Yüzde Değişim"]),
            })
    return rows

def _stats(rows: List[Dict]) -> Dict:
    if not rows: return {"count":0,"avgNew":0.0,"avgDelta":0.0,"improved":0}
    n=len(rows)
    avgNew=sum((r.get("newScore") or 0) for r in rows)/n
    avgDelta=sum((r.get("delta") or 0) for r in rows)/n
    improved=sum(1 for r in rows if (r.get("delta") or 0)>0.001)
    return {"count":n,"avgNew":avgNew,"avgDelta":avgDelta,"improved":improved}

@app.get("/api/results/{kind}")
def get_results(kind: Literal["niyet","sorgu"]):
    path = NIYET_CSV if kind=="niyet" else SORGU_CSV
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"Çıktı yok: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    rows = _normalize(df, kind)
    return {"rows": rows, "stats": _stats(rows), "path": path}

@app.get("/api/ping")
def ping(): return {"ok": True}
