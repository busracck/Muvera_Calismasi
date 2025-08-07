import os
import json
import re
import pandas as pd
from ollama import Client
from sentence_transformers import SentenceTransformer, util
from prompts.niyet_prompt import get_niyet_iyilestirme_prompt
import logging

# Logger ayarı
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Bağlantılar
ollama_client = Client(host="http://localhost:11434")
model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

def hesapla_benzerlik(sorgu, gelistirilmis_metin):
    logger.debug("Benzerlik hesaplanıyor")
    sorgu_emb = model.encode(sorgu, convert_to_tensor=True)
    metin_emb = model.encode(gelistirilmis_metin, convert_to_tensor=True)
    skor = util.cos_sim(sorgu_emb, metin_emb)
    logger.debug(f"Benzerlik skoru: {skor.item()}")
    return round(float(skor), 4)

def temizle_raw_output(text):
    logger.debug("Ham model çıktısı:\n%s", text)
    try:
        text = text.strip().replace("```json", "").replace("```", "").strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            logger.warning("Süslü parantez bulunamadı")
            return {}

        json_candidate = text[json_start:json_end].strip()
        logger.debug("JSON adayı:\n%s", json_candidate)

        parsed = json.loads(json_candidate)
        logger.debug("JSON başarıyla parse edildi")
        return parsed
    except Exception as e:
        logger.exception("temizle_raw_output içinde hata")
        return {}

def gelistir_icerik(sorgu, mevcut_metin, html_bolumu):
    logger.info("İçerik geliştiriliyor: %s", sorgu)
    try:
        prompt_template = get_niyet_iyilestirme_prompt(sorgu, mevcut_metin, html_bolumu)  
        messages = prompt_template.format_messages(
            sorgu=sorgu,
            mevcut_icerik=mevcut_metin,
            html_bolumu=html_bolumu
        )
        ollama_messages = [
            {"role": "user", "content": f"{m.content}"} if m.type == "human" else
            {"role": "assistant", "content": f"{m.content}"} if m.type == "ai" else
            {"role": "system", "content": f"{m.content}"} for m in messages
        ]

        response = ollama_client.chat(
            model="gemma:instruct",
            messages=ollama_messages,
            options={"temperature": 0.2}
        )

        raw_output = response.get("message", {}).get("content", None)
        logger.debug("Model yanıtı:\n%s", raw_output)

        if raw_output is None:
            logger.warning("Modelden hiç yanıt gelmedi")
            raise RuntimeError("Modelden yanıt alınamadı")

        yanit = temizle_raw_output(raw_output)
        if not yanit or not isinstance(yanit, dict):
            logger.warning("Yanıt boş veya dict formatında değil")
            return {}

        return {
            "sorgu": yanit.get("sorgu", sorgu),
            "mevcut_icerik": yanit.get("mevcut_icerik", mevcut_metin),
            "gelistirilmis_icerik": yanit.get("gelistirilmis_icerik", mevcut_metin).strip()
        }
    except Exception as e:
        logger.exception("gelistir_icerik fonksiyonunda hata")
        return {}

def zincirleme_gelistirme(sorgu, icerik, html, max_steps=3):
    logger.info("Zincirleme geliştirme başlıyor")
    mevcut = icerik
    for adim in range(max_steps):
        logger.debug("Adım %d: %s", adim + 1, mevcut)
        yanit = gelistir_icerik(sorgu, mevcut, html)
        if not yanit:
            logger.warning("Model yanıtı boş, zincir durdu")
            break
        yeni = yanit.get("gelistirilmis_icerik", "").strip()
        if not yeni or yeni == mevcut:
            logger.info("İçerikte gelişme yok, zincir durdu")
            break
        mevcut = yeni
    return mevcut

logger.info("CSV dosyası okunuyor")
df = pd.read_csv("html_icerik_niyet_uyumu.csv")
hedefler = df[
    (df["Uyum Durumu"] == "uyumlu") &
    (df["Benzerlik Skoru"] > 0.65) &
    (df["Benzerlik Skoru"] < 0.85)
]
logger.info("Geliştirilecek içerik sayısı: %d", len(hedefler))

os.makedirs("output", exist_ok=True)
sonuclar = []

for index, row in hedefler.iterrows():
    sorgu = row["Kullanıcı Niyeti"]
    mevcut = row["İçerik"]
    html = row["HTML Bölümü"]
    eski_skor = row["Benzerlik Skoru"]
    logger.info("İşleniyor: %d - %s", index + 1, sorgu)

    try:
        gelistirilmis = zincirleme_gelistirme(sorgu, mevcut, html)
        if not gelistirilmis:
            gelistirilmis = mevcut

        yeni_skor = hesapla_benzerlik(sorgu, gelistirilmis)
        yuzde_degisim = round(((yeni_skor - eski_skor) / eski_skor) * 100, 2)
        degisim_ifadesi = f"%{yuzde_degisim} artmış" if yuzde_degisim >= 0 else f"%{abs(yuzde_degisim)} azalmış"

        logger.info("Sonuç eklendi (%s)", degisim_ifadesi)
        sonuclar.append({
            "sorgu": sorgu,
            "mevcut_icerik": mevcut,
            "gelistirilmis_icerik": gelistirilmis,
            "HTML Bölümü": html,
            "Eski Skor": eski_skor,
            "Yeni Skor": yeni_skor,
            "Yüzde Değişim": degisim_ifadesi
        })

    except Exception as e:
        logger.exception("Hata oluştu: %s", sorgu)

output_path = "output/daha_uyumlu_json_formatli_output.csv"
pd.DataFrame(sonuclar).to_csv(output_path, index=False, encoding="utf-8-sig")
logger.info("Kaydedildi → %s", output_path)
logger.info("Tüm işlemler tamamlandı")
