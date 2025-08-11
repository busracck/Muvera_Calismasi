from langchain_core.prompts import ChatPromptTemplate

def build_prompt(kullanici_niyeti: str, mevcut_icerik: str, html_bolumu: str, eski_skor: float) -> str:
    """
    LLM'ye, metni küçük dokunuşlarla geliştirmesi için net talimat verir.
    Hedef çıktı: JSON formatında eksiksiz döndürülmesi.
    """
    
    # System Template
    system_template = """
Sen bir SEO ve içerik geliştirme uzmanısın.
Görevin, kullanıcı niyetine (intent) göre mevcut metni küçük dokunuşlarla iyileştirmektir.

Kurallar:
1. Benzerlik skorunu artırmaya odaklan, anlamı bozma.
2. Metin türüne göre:
   - h1/h2: Kullanıcı niyetinden DOĞRUDAN başlık üret (örn: "google reklam verme nasıl" → "Google Reklam Verme Nasıl Yapılır?").
   - p/div: Mevcut metni TAMAMEN KORU, sadece kullanıcı niyetini karşılayacak şekilde en fazla 5-10 kelime ekle. Örneğin:
     * Mevcut: "Google reklamları hızlı ve pratiktir."
     * Niyet: "google reklam verme nasıl"
     * Geliştirilmiş: "Google reklamları hızlı ve pratiktir; google reklam verme nasıl yapılır sorusuna kısa bir yanıt sunar."
   - li: Mevcut metni TAMAMEN KORU, sadece kullanıcı niyetini karşılayacak şekilde en fazla 1-2 kelime ekle.
3. Uzunluk sınırları:
   - p/div: Mevcut metne en fazla 5-10 kelime ekle.
   - li: Mevcut metne en fazla 1-2 kelime ekle.
   - h1/h2: "Nasıl Yapılır?" veya "Kılavuzu" varsa ASLA kesilmez.
4. Her zaman değiştir. Rollback yok.
5. Reklam/CTA/pazarlama ifadeleri kullanma: "hedef kitlenize ulaşın", "kampanyanızı oluşturun", "potansiyel müşteriler" vb. yasak.
6. Cevap her zaman geçerli JSON olmalı.
7. KRİTİK KURAL: Mevcut metin mutlaka korunmalı, sadece küçük eklemeler yapılmalı. Metni kısaltma veya özetleme yapma!
8. ÖNEMLİ: Kullanıcı niyetini doğrudan karşılayan ifadeler ekle (örn: "google reklam verme nasıl" için "nasıl yapılır" ifadesi mutlaka geçmeli).
"""
    
    # Human Template
    human_template = """
Girdi:
Kullanıcı Niyeti: "{kullanici_niyeti}"
Mevcut İçerik: "{mevcut_icerik}"
HTML Bölümü: "{html_bolumu}"
Eski Skor: {eski_skor}

Beklenen çıktı formatı (JSON):
{{
  "Kullanıcı Niyeti": "{kullanici_niyeti}",
  "Mevcut İçerik": "{mevcut_icerik}",
  "Geliştirilmiş İçerik": "Buraya geliştirilmiş hali gelecek",
  "HTML Bölümü": "{html_bolumu}"
}}

Sadece bu JSON'u döndür, başka açıklama veya metin ekleme.
"""
    
    # ChatPromptTemplate oluştur
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    # Formatla ve döndür
    formatted_prompt = prompt.format(
        kullanici_niyeti=kullanici_niyeti,
        mevcut_icerik=mevcut_icerik,
        html_bolumu=html_bolumu,
        eski_skor=eski_skor
    )
    
    return formatted_prompt