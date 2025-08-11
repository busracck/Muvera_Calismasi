# -*- coding: utf-8 -*-
"""
Amaç: Kullanıcı Sorgusu ile Eski Metin uyumunu küçük ve güvenli dokunuşlarla artıran KISA bir metin üretmek.
Çıktı: Tek JSON.
"""

SYSTEM_TEMPLATE = """
Rolün: Türkçe içerik editörü (E-E-A-T & semantik SEO).
Kurallar:
- Anlamı koru, yeni bilgi uydurma, CTA/pazarlama ekleme.
- h1/h2: Tek cümle, nokta yok; soruysa "?".
- li: Tek cümle, sonda nokta.
- p/div: En fazla 2 kısa cümle.
- Niyet terimi: Sorgudaki ana terim en az 1 kez birebir geçsin.
- SADECE geçerli JSON döndür.
JSON Şema:
{
  "kullanici_sorgusu": "...",
  "eski_metin": "...",
  "kisa_duzenleme": "..."
}
""".strip()

HUMAN_TEMPLATE = """
Kullanıcı Sorgusu: {kullanici_sorgusu}
Eski Metin: {eski_metin}
HTML Bölümü: {html_bolumu}
Eski Skor: {eski_skor}

Lütfen SADECE geçerli bir JSON döndür.
""".strip()

def build_prompt(kullanici_sorgusu: str, eski_metin: str, html_bolumu: str, eski_skor: float) -> str:
    system = f"<|system|>\n{SYSTEM_TEMPLATE}"
    user = f"<|user|>\n" + HUMAN_TEMPLATE.format(
        kullanici_sorgusu=kullanici_sorgusu,
        eski_metin=eski_metin,
        html_bolumu=html_bolumu,
        eski_skor=eski_skor,
    )
    return system + "\n" + user
