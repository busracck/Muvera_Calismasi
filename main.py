from anlamsal_eslestirme import anlamsal_eslestirme, tam_niyet_uyum_tablosu, tam_sorgu_uyum_tablosu # type: ignore
from intent_classifier import niyet_belirle
from anlamsal_eslestirme import title_description_uyumu   # type: ignore
from anlamsal_eslestirme import title_description_birbirine_uyum # type: ignore
from kullanici_sorgusu import sorgular
from webScraping import get_structured_web_content_selenium
import pandas as pd # type: ignore
import re

def temizle_niyet(text):
    if not text:
        return ""
    
    text = text.lower().strip()
    text = re.sub(r"[.?!,:;]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace('"', '').replace("'", '')
    return text

# ----------------------------- #
# 1. URL input
# ----------------------------- #
url = input("Analiz edilecek web sayfası URL'si: ").strip()

# ----------------------------- #
# 2. Anlamsal eşleşmeler
# ----------------------------- #
print("\n🔍 Anlamsal eşleşmeler yapılıyor...")
eslesme_df = anlamsal_eslestirme(url)

# ----------------------------- #
# 3. Kullanıcı niyeti tahmini
# ----------------------------- #
print("\n🧠 Kullanıcı niyetleri çıkarılıyor...")
niyetler = []
for sorgu in eslesme_df["Sorgu"]:
    niyet = niyet_belirle(sorgu)
    print(f"{sorgu} → {niyet}")
    niyetler.append(niyet)

eslesme_df["Kullanıcı Niyeti"] = [temizle_niyet(n) for n in niyetler]

# ----------------------------- #
# 5. Sayfa içeriğini getir
# ----------------------------- #
content = get_structured_web_content_selenium(url)
niyet_listesi = eslesme_df["Kullanıcı Niyeti"].unique().tolist()

# ----------------------------- #
# 6. Tüm içerik × niyet analizi
# ----------------------------- #
print("\n📊 Tüm içerik ve niyetler ayrıntılı olarak eşleştiriliyor...")
tam_niyet_df = tam_niyet_uyum_tablosu(content, niyet_listesi)
tam_niyet_df.to_csv("html_icerik_niyet_uyumu.csv", index=False)
print("✅ Detaylı içerik-niyet eşleşme sonucu 'html_icerik_niyet_uyumu.csv' dosyasına kaydedildi.")
print(tam_niyet_df.head())

# ----------------------------- #
# 7. Tüm içerik × sorgu analizi
# ----------------------------- #
print("\n📊 Tüm içerik ve sorgular ayrıntılı olarak eşleştiriliyor...")
tam_sorgu_df = tam_sorgu_uyum_tablosu(content, sorgular)
tam_sorgu_df.to_csv("html_icerik_sorgu_uyumu.csv", index=False)
print("✅ Detaylı içerik-sorgu eşleşme sonucu 'html_icerik_sorgu_uyumu.csv' dosyasına kaydedildi.")
print(tam_sorgu_df.head())

# ----------------------------- #
# 8. Title ve Description Kullanıcı Sorgusuna Göre Uyumu
# ----------------------------- #
print("\n📝 Başlık ve açıklama alanları sorgularla karşılaştırılıyor...")
title_desc_df = title_description_uyumu(content, sorgular)
title_desc_df.to_csv("title_description_uyum.csv", index=False)
print("✅ 'title_description_uyum.csv' dosyasına yazıldı.")
print(title_desc_df.head())


# ----------------------------- #
# 9. Başlık ve açıklama birbirine göre uyumu
# ----------------------------- #
print("\n📊 Başlık ve açıklamanın birbirine göre anlamsal uyumu ölçülüyor...")
title_meta_df = title_description_birbirine_uyum(content)
title_meta_df.to_csv("title_description_kendi_uyumu.csv", index=False)
print("✅ 'title_description_kendi_uyumu.csv' dosyasına yazıldı.")
print(title_meta_df)