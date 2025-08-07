from ollama import Client
import pandas as pd
from kullanici_sorgusu import sorgular
import re

ollama_client = Client(host='http://localhost:11434')  # Ollama arka planda çalışmalı

def niyet_belirle(sorgu):
    prompt = f"""
Bir kullanıcı şu arama sorgusunu yazdı: "{sorgu}"

Bu sorgunun özünde hangi amaç yatıyor?

Lütfen yalnızca 3–5 kelimelik, sade ve tematik bir niyet ifadesi ver.
Bir etiket ya da başlık gibi düşün. Nokta veya açıklama yazma.
"""
    response = ollama_client.chat(
        model='gemma3:4b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content'].strip().lower()

# Tüm sorgular için çalıştır
sonuclar = []
for sorgu in sorgular:
    niyet = niyet_belirle(sorgu)
    print(f"{sorgu} → {niyet}")
    sonuclar.append({"Sorgu": sorgu, "Kısa Niyet Teması": niyet})

# CSV'ye yaz
df = pd.DataFrame(sonuclar)
df.to_csv("sorgu_niyet_tema.csv", index=False)
