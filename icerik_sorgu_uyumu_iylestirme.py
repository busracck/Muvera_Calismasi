from ollama import Client
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from prompts.sorgu_prompt import get_sorgu_iyilestirme_prompt

ollama_client = Client(host="http://localhost:11434")
model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

def hesapla_benzerlik(sorgu, gelistirilmis_metin):
    sorgu_emb = model.encode(sorgu, convert_to_tensor=True)
    metin_emb = model.encode(gelistirilmis_metin, convert_to_tensor=True)
    skor = util.cos_sim(sorgu_emb, metin_emb)
    return round(float(skor), 4)

def gelistir_icerik(sorgu, mevcut_metin, html_bolumu):
    prompt = get_sorgu_iyilestirme_prompt()
    messages = prompt.format_messages(
        sorgu=sorgu,
        mevcut_metin=mevcut_metin,
        html_bolumu=html_bolumu
    )
    response = ollama_client.chat(
        model="gemma3:4b",
        messages=[m.dict() for m in messages],
        options={'temperature': 0.2}
    )
    return response["message"]["content"]

df = pd.read_csv("html_icerik_sorgu_uyumu.csv")
hedefler = df[
    (df["Uyum Durumu"] == "uyumlu") &
    (df["Benzerlik Skoru"] > 0.65) &
    (df["Benzerlik Skoru"] < 0.85)
]

print(f"Toplam geliştirilecek içerik sayısı: {len(hedefler)}")
sonuclar = []

for _, row in hedefler.iterrows():
    sorgu = row["Kullanıcı Sorgusu"]
    mevcut = row["İçerik"]
    html = row["HTML Bölümü"]
    eski_skor = row["Benzerlik Skoru"]

    try:
        gelistirilmis = gelistir_icerik(sorgu, mevcut, html)
        yeni_skor = hesapla_benzerlik(sorgu, gelistirilmis)
        yuzde_degisim = round(((yeni_skor - eski_skor) / eski_skor) * 100, 2)
        degisim_ifadesi = f"%{yuzde_degisim} artmış" if yuzde_degisim >= 0 else f"%{abs(yuzde_degisim)} azalmış"

        sonuclar.append({
            "HTML Bölümü": html,
            "Sorgu": sorgu,
            "Eski Metin": mevcut,
            "Geliştirilmiş Metin": gelistirilmis,
            "Eski Skor": eski_skor,
            "Yeni Skor": yeni_skor,
            "Yüzde Değişim": degisim_ifadesi
        })

    except Exception as e:
        print(f"Hata oluştu: '{sorgu}' → {e}")

pd.DataFrame(sonuclar).to_csv("daha_uyumlu_sorgu_oneriler_yuzde_degisim.csv", index=False, encoding="utf-8-sig")
print("✅ Geliştirme tamamlandı.")