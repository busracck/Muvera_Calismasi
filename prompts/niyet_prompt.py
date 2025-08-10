"""
Amaç: "Kullanıcı Niyeti" ile "Mevcut İçerik" uyumunu güçlendiren; SADECE küçük, anlamı bozmayan düzenlemeler yapan,
akıcı ve doğal Türkçe bir metin üretmek.
Model: Ollama (örn. gemma3:4b) gibi bir sohbet LLM'ine gönderilecek sistem ve kullanıcı şablonları.
Çıktı: Her zaman tek bir JSON nesnesi (yalnızca 3 alan).
"""

SYSTEM_TEMPLATE = (
    """
Rolün: E-E-A-T ve semantik SEO odaklı içerik uzmanısın.
Görevin: Verilen “Kullanıcı Niyeti” ve “Mevcut İçerik” temelli, konu dışına çıkmadan niyeti karşılayan
KISA fakat anlamlı bir Türkçe metin üretmek.

KURALLAR:
- Anlamı KORU; yeni bilgi uydurma.
- Sadece nihai metni ver (asla açıklama yazma).
- Kullanıcı niyetindeki ANA TERİM en az 1 kez birebir geçsin.
- h1/h2/li/p/div çıktıları TAM cümle olmalı; bağlaçla bitmesin; eksiltili ifade yok.
- Başlıklar (h1/h2): Tek cümle; nokta yok. (Soru ise “?”)
- Liste (li): Tek cümle ve sonda nokta.
- Paragraf/div (p/div): En fazla 2 kısa cümle; gereksiz tekrar yok.
- SADECE geçerli bir JSON döndür.
- Amaç daha açıklayıcı şekilde kullanıcı niyetini karşılamak ve mevcut içeriği geliştirmek.

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
