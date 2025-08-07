from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_niyet_iyilestirme_prompt(sorgu, mevcut_metin, html_bolumu):
    system_template = """
Sen bir SEO ve içerik yazarlığı uzmanısın.

Kullanıcının amacı, Google reklamları hakkında bilgi edinmektir.

Aşağıda bir sorgu, buna ait mevcut içerik ve HTML etiketi verilmiştir. Mevcut içeriği geliştir.

Kurallar:
- Sorguya tam ve net yanıt ver.
- Açıklayıcı, öğretici ve özgün içerik üret.
- Gerekiyorsa kısa örnek veya liste ekle.
- SEO ve E-E-A-T ilkelerine dikkat et.

Sadece şu formatta JSON dön:
```json
{
    "sorgu": "Sorgu buraya",
    "mevcut_icerik": "Eski içerik buraya",
    "gelistirilmis_icerik": "Yeni geliştirilmiş içerik buraya"
}

```
    """ 

    human_template = """
Kullanıcı Sorgusu: {sorgu}

Mevcut İçerik: {mevcut_icerik}

HTML Bölümü: {html_bolumu}
    """

    system_prompt = SystemMessagePromptTemplate.from_template(system_template.strip())
    human_prompt = HumanMessagePromptTemplate.from_template(human_template.strip())

    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
