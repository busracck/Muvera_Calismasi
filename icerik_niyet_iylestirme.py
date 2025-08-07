import json
import pandas as pd
import re
from ollama import Client
from sentence_transformers import SentenceTransformer, util
from prompts.niyet_prompt import get_niyet_iyilestirme_prompt

# Ollama ve model
ollama_client = Client(host="http://localhost:11434")
model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

def hesapla_benzerlik(sorgu, gelistirilmis_metin):
    sorgu_emb = model.encode(sorgu, convert_to_tensor=True)
    metin_emb = model.encode(gelistirilmis_metin, convert_to_tensor=True)
    skor = util.cos_sim(sorgu_emb, metin_emb)
    return round(float(skor), 4)

def temizle_raw_output(text):
    try:
        # Kod bloğu başlıklarını temizle
        text = text.strip().replace("```json", "").replace("```", "").strip()

        # JSON gibi başlayan kısmı al
        match = re.search(r'{.*}', text, re.DOTALL)
        if match:
            cleaned = match.group(0)
            # Burada parse deneyelim
            json.loads(cleaned)  # Valid mi kontrolü
            return cleaned
        return "{}"

    except Exception as e:
        print(f"❌ temizle_raw_output hata: {e}")
        print("⛔ Ham model çıktısı:\n", text)
        return "{}"



def gelistir_icerik(sorgu, mevcut_metin, html_bolumu):
    prompt_template = get_niyet_iyilestirme_prompt(sorgu, mevcut_metin, html_bolumu)  
    messages = prompt_template.format_messages(
        sorgu=sorgu,
        mevcut_icerik=mevcut_metin,
        html_bolumu=html_bolumu
    )

    ollama_messages = [{"role": m.type, "content": m.content} for m in messages]

    try:
        response = ollama_client.chat(
            model="gemma:instruct",
            messages=ollama_messages,
            options={"temperature": 0.2}
        )

        if not response or "message" not in response or "content" not in response["message"]:
            print(f"⚠️ Model boş yanıt verdi: {response}")
            return {}

        raw_output = response["message"]["content"]
        print("\n🔴 MODEL YANITI:\n", raw_output)

        json_candidate = temizle_raw_output(raw_output)

        try:
            yanit = json.loads(json_candidate)
        except Exception as e:
            print(f"⚠️ JSON parse hatası: {e}")
            return {}

        return {
            "sorgu": yanit.get("sorgu", sorgu),
            "mevcut_icerik": yanit.get("mevcut_icerik", mevcut_metin),
            "gelistirilmis_icerik": yanit.get("gelistirilmis_icerik", "").strip()
        }

    except Exception as e:
        print(f"❌ HATA (chat/parse): {e}")
        return {}

# CSV'den oku
df = pd.read_csv("html_icerik_niyet_uyumu.csv")

# Hedefleri filtrele
hedefler = df[
    (df["Uyum Durumu"] == "uyumlu") &
    (df["Benzerlik Skoru"] > 0.65) &
    (df["Benzerlik Skoru"] < 0.85)
]

print(f"🎯 Geliştirilecek içerik sayısı: {len(hedefler)}")
sonuclar = []

for _, row in hedefler.iterrows():
    sorgu = row["Kullanıcı Niyeti"]
    mevcut = row["İçerik"]
    html = row["HTML Bölümü"]
    eski_skor = row["Benzerlik Skoru"]

    try:
        yanit = gelistir_icerik(sorgu, mevcut, html)
        if not yanit:
            raise ValueError("Model çıktısı boş.")

        gelistirilmis = yanit.get("gelistirilmis_icerik", "").strip()
        if not gelistirilmis:
            raise ValueError("Geliştirilmiş içerik boş.")

        yeni_skor = hesapla_benzerlik(sorgu, gelistirilmis)
        yuzde_degisim = round(((yeni_skor - eski_skor) / eski_skor) * 100, 2)
        degisim_ifadesi = f"%{yuzde_degisim} artmış" if yuzde_degisim >= 0 else f"%{abs(yuzde_degisim)} azalmış"

        sonuclar.append({
            "sorgu": yanit.get("sorgu", sorgu),
            "mevcut_icerik": yanit.get("mevcut_icerik", mevcut),
            "gelistirilmis_icerik": gelistirilmis,
            "HTML Bölümü": html,
            "Eski Skor": eski_skor,
            "Yeni Skor": yeni_skor,
            "Yüzde Değişim": degisim_ifadesi
        })

    except Exception as e:
        print(f"❌ Hata oluştu → '{sorgu}': {e}")

# Kaydet
pd.DataFrame(sonuclar).to_csv("output/daha_uyumlu_json_formatli_output.csv", index=False, encoding="utf-8-sig")
print("✅ Tüm işlemler tamamlandı.")