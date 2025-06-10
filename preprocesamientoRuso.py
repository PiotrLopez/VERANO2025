#Libs
import numpy as np
import pandas as pd
import re
import csv
import nltk
import string
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from cleantext import clean
from collections import defaultdict
nltk.download('stopwords')
toktok = ToktokTokenizer()
stop_words = set(stopwords.words('english'))

#Preprocesamiento

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def remove_punctuation_mark(text, replace=" "):
    text = text.replace("\n", " ")
    return re.sub(r'[%s]' % re.escape(string.punctuation + '¡¿´©✕“”‘’†•−˚'), replace, text)

def tokens(text):
    return toktok.tokenize(text)

def avoidVoid(text):
    return isinstance(text, str) and text.strip() != ""

def get_bigrams(words):
    return " ".join([f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)])

def preprocess(df, minrep=1, bigramas=0, max_doc_freq=0.7):
    new_df = []
    list_df = []
    doc_freq = defaultdict(int)
    total_docs = 0

    for text in df:
        if avoidVoid(text):
            text = clean(text, no_emoji=True)
            text = text.lower()
            text = remove_stop_words(text)
            text = remove_punctuation_mark(text)
            words = tokens(text)

            if bigramas == 0:
                processed_text = " ".join(words)
            elif bigramas == 1:
                processed_text = get_bigrams(words)
            elif bigramas == 2:
                processed_text = " ".join(words) + " " + get_bigrams(words)

            list_df.append(processed_text)
            unique_words = set(processed_text.split())
            for word in unique_words:
                doc_freq[word] += 1
            total_docs += 1

    common_words = {word for word, freq in doc_freq.items() if freq / total_docs > max_doc_freq}

    vocabulario = []
    for text in list_df:
        filtered_text = " ".join([word for word in text.split() if word not in common_words])
        new_df.append(filtered_text)
        vocabulario.extend(filtered_text.split())

    vocabulario = sorted(set(vocabulario))
    return new_df, vocabulario

def remove_repeated_ngrams(text, max_ngram_size=5):
    for n in range(max_ngram_size, 1, -1):
        pattern = re.compile(rf'(\b(?:\w+)\b(?:\s+\b(?:\w+)\b){{{n-1}}})(?:\s+\1)+', flags=re.IGNORECASE)
        while True:
            new_text = pattern.sub(r'\1', text)
            if new_text == text:
                break
            text = new_text
    return text

global_path = r'C:\Users\piotr\Desktop\VERANO2025\Descripciones'
coleccion = 'MAZATLAN_desc.csv'
file_path = f'{global_path}/{coleccion}'
text_df = pd.read_csv(file_path)
raw_texts = text_df['prompt'].tolist()
# Usar un umbral de frecuencia relativa 
processed_texts, vocabulario = preprocess(raw_texts, bigramas=0, max_doc_freq=0.05)
processed_texts = [remove_repeated_ngrams(text) for text in processed_texts]
text_df['prompt'] = processed_texts
save_path = 'MAZATLAN_desc_preprocesado.csv'
text_df.to_csv(save_path, sep=',', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
print("Archivo guardado en:", save_path)
