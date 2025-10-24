import re

def clean_the_text(original_text):

    if not isinstance(original_text, str):#string değilse boş döner
        return ""


    metin = original_text.lower()
    metin = re.sub(r'http\S+|www\S+', '', metin)#url kaldırır
    metin = re.sub(r'<.*?>', '', metin)#etiket kaldırır (html)
    metin = re.sub(r'\d+', '', metin)#sayıları kaldırır
    metin = re.sub(r'[^\w\s]', '', metin)#noktalamaları kaldırır
    metin = re.sub(r'\s+', ' ', metin).strip()#birden fazla boşluğu bire indirger
    return metin