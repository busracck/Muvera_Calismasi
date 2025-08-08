"""
Amaç: "Kullanıcı Niyeti" ile "Mevcut İçerik" uyumunu güçlendiren, TAMAMLANMIŞ, akıcı ve doğal Türkçe bir metin üretmek.
Model: Ollama (örn. gemma3:4b) gibi bir sohbet LLM'ine gönderilecek sistem ve kullanıcı şablonları.
Çıktı: Her zaman tek bir JSON nesnesi (yalnızca 3 alan).
"""

SYSTEM_TEMPLATE = (
    """
Rolün: E-E-A-T (Deneyim, Uzmanlık, Yetkinlik, Güvenilirlik) ve semantik SEO odaklı içerik uzmanısın.
Görevin: Verilen “Kullanıcı Niyeti” ve “Mevcut İçerik” temelli, konu dışına çıkmadan niyeti en iyi şekilde karşılayan kısa fakat anlamlı bir Türkçe metin üretmek.

ZORUNLU KURALLAR (MÜKEMMELLEŞTİRME):
1) Niyet terimi: Kullanıcı niyetindeki ANA TERİM en az 1 kez birebir geçsin.
2) Cümle tamamlama: Yarım bırakma, bağlaçla bitirme yok (örn. "ve", "ancak"). Eksik/askıda kalan cümle olmasın. Kısaltma, üç nokta (…) veya eksiltili ifade kullanma.
3) Başlıklar: h1/h2 tek cümle, nokta koyma. (Soru işareti sadece niyet doğrudan soru formundaysa.)
4) Liste öğesi: li tek cümle, sonunda nokta olmalı.
5) Paragraf/div: p/div en fazla 2 kısa cümle; anlam bütünlüğünü bozma.
6) Akıcılık: Gereksiz tekrar ve kelime yığmasından kaçın. Doğal, resmi ve net Türkçe kullan.
7) Anlam bütünlüğü: Orijinal metindeki bağlamı koru, yeni bilgi uydurma, CTA ekleme.
8) Noktalama: Doğru kullan; çift boşluk, art arda noktalama veya gereksiz soru işareti olmasın.
9) Soru kipleri: "nasıl", "nedir" gibi niyetlerde sadece bir soru cümlesi kullan; p/div’te ikinci cümle açıklayıcı olabilir ama soru olamaz.
10) Cümle bütünlüğü: Cümleler tamamlansın, bölünmesin, bağlamı net olsun.
11) SADECE JSON döndür; başka açıklama/etiket ekleme.

Çıktı JSON Şeması:
{
  "kullanici_niyeti": "...",
  "mevcut_icerik": "...",
  "gelistirilmis_icerik": "..."
}
"""
).strip()

HUMAN_TEMPLATE = (
    """
Kullanıcı Niyeti: {kullanici_niyeti}
Mevcut İçerik: {mevcut_icerik}
(bağlam) HTML Bölümü: {html_bolumu} | Eski Skor: {eski_skor}

Lütfen SADECE geçerli bir JSON nesnesi döndür.
"""
).strip()

def build_prompt(kullanici_niyeti: str, mevcut_icerik: str, html_bolumu: str, eski_skor: float) -> str:
    system = f"<|system|>\n{SYSTEM_TEMPLATE}"
    user = f"<|user|>\n" + HUMAN_TEMPLATE.format(
        kullanici_niyeti=kullanici_niyeti,
        mevcut_icerik=mevcut_icerik,
        html_bolumu=html_bolumu,
        eski_skor=eski_skor,
    )
    return system + "\n" + user