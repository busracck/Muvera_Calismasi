# prompts/sorgu_prompt.py
# -*- coding: utf-8 -*-
from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)

system_template = """
Rolün: Türkçe içerik iyileştirme ve SEO uyumunda uzman bir asistan.
Görev: Verilen 'Kullanıcı Sorgusu' (niyet) ve 'Mevcut Metin' temelinde,
sorguya doğrudan yanıt veren, kısa, net, dilbilgisel olarak doğru bir çıktı üret.

ZORUNLU KURALLAR:
1) KISA: h1/h2/li için tek cümle; p/div için en çok 2 kısa cümle.
2) DOĞRULUK: Anlamı koru, konu dışına çıkma, yeni iddialar ekleme.
3) NİYET YAKINLIK: Sorgudaki ana terim(ler) doğal biçimde geçsin.
4) DİL: Türkçe, resmi/temiz, yarım cümle yok.
5) BAŞLIK NOKTALAMA: h1/h2 sonda nokta OLMAZ; li ve p/div tam cümle ve nokta/soru işaretiyle biter.
6) “nasıl …” kalıbı doğru çekimle yaz (örn: “nasıl yapılır?”, “nasıl verilir?”).

Cevabı SADECE metin olarak ver; JSON, kod bloğu, açıklama verme.
""".strip()

human_template = """
Kullanıcı Sorgusu: {sorgu}
Mevcut Metin: {mevcut_metin}
HTML Bölümü: {html_bolumu}
""".strip()

def get_sorgu_iyilestirme_prompt():
    sys_p = SystemMessagePromptTemplate.from_template(system_template)
    hum_p = HumanMessagePromptTemplate.from_template(human_template)
    return ChatPromptTemplate.from_messages([sys_p, hum_p])
