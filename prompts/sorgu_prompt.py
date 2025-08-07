from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore

def get_sorgu_iyilestirme_prompt():
    system_template = """
    Kullanıcı şu sorguyla geldi: "{sorgu}"

Mevcut içerik bu sorguyla %65-%85 benzerliğe sahip. Senin görevin:
- Bu içeriği anlamını bozmadan yeniden yazmak.
- Başta kullanıcı sorgusunu doğrudan yanıtla.
- Sonra mevcut metnin içeriklerine sadık kalarak uygun detayları ekle.
- Gereksiz tekrarlar, konu dışı ifadeler olmasın.
- Yeni metin 1-2 kısa paragraf olmalı.
- Sadece geliştirilen yeni metni döndür. HTML etiketi, başlık veya açıklama verme.
    """

    human_template = """
    Sorgu: {sorgu}
    Mevcut İçerik: {mevcut_metin}
    HTML Bölümü: {html_bolumu}
    """

    system_prompt = SystemMessagePromptTemplate.from_template(system_template.strip())
    human_prompt = HumanMessagePromptTemplate.from_template(human_template.strip())

    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])