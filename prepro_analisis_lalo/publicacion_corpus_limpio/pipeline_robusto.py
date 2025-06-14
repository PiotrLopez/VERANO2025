import pandas as pd
import numpy as np
import spacy
from unidecode import unidecode
import re
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing as mp

# Configuración avanzada
nlp = spacy.load("es_core_news_lg", disable=["parser", "ner"])
stopwords = nlp.Defaults.stop_words
custom_stopwords = {"ser", "estar", "haber", "tener", "hacer", "poder", "decir", "si", "sí", "vez", "tan", "mas", "más"}
stopwords = stopwords.union(custom_stopwords)

# Función de limpieza mejorada
def clean_text_advanced(text):
    if not isinstance(text, str):
        return ""
    
    # Normalización avanzada
    text = unidecode(text).lower()
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^\w\s]|_', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Procesamiento lingüístico
    doc = nlp(text)
    cleaned_tokens = []
    
    for token in doc:
        if (token.text not in stopwords and 
            len(token.text) > 2 and 
            token.pos_ not in ["PRON", "DET", "AUX"] and
            token.lemma_ not in ["ser", "estar"]):
            
            # Lematización selectiva
            if token.pos_ in ["VERB", "NOUN", "ADJ"]:
                cleaned_tokens.append(token.lemma_)
            else:
                cleaned_tokens.append(token.text)
    
    # Unificar texto y eliminar redundancias
    text = ' '.join(cleaned_tokens)
    text = re.sub(r'\b(\w+)\b(?=.*\b\1\b)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Generar embeddings semánticos
def get_embeddings(texts):
    embeddings = []
    for doc in tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="Generando embeddings"):
        embeddings.append(doc.vector)
    return np.array(embeddings)

# Encontrar palabras clave por cluster
def get_cluster_keywords(texts, clusters, n_keywords=10):
    cluster_keywords = {}
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster]
        
        # Crear TF-IDF solo para este cluster
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        
        # Calcular importancia promedio de términos
        avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = avg_tfidf.argsort()[-n_keywords:][::-1]
        
        keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        cluster_keywords[cluster] = keywords
    
    return cluster_keywords

# Procesar un archivo individual
def process_file(file_path, output_folder, n_clusters=8):
    # Leer archivo
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    city_name = filename.replace('.csv', '').upper()
    
    print(f"\nProcesando: {city_name}")
    
    # Preprocesar textos
    texts = [clean_text_advanced(t) for t in tqdm(df['texto_limpio'].astype(str), desc="Limpiando textos")]
    
    # Filtrar textos vacíos
    valid_texts = []
    valid_indices = []
    for i, t in enumerate(texts):
        if len(t.split()) > 3:  # Filtrar textos con menos de 4 palabras
            valid_texts.append(t)
            valid_indices.append(i)
    
    # Obtener embeddings semánticos
    if not valid_texts:
        print(f"⚠️ No hay textos válidos en {filename}. Saltando...")
        return
    
    embeddings = get_embeddings(valid_texts)
    
    # Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Obtener palabras clave por cluster
    cluster_keywords = get_cluster_keywords(valid_texts, clusters)
    
    # Crear DataFrame final para este archivo
    final_df = df.iloc[valid_indices].copy()
    final_df['texto_procesado'] = valid_texts
    final_df['cluster'] = clusters
    final_df['tema_principal'] = [cluster_keywords[c][0] for c in clusters]
    final_df['ciudad'] = city_name
    
    # Crear resumen de clusters
    cluster_summary = []
    for cluster_id, keywords in cluster_keywords.items():
        cluster_size = np.sum(clusters == cluster_id)
        cluster_summary.append({
            'ciudad': city_name,
            'cluster': cluster_id,
            'tamano': cluster_size,
            'palabras_clave': ', '.join(keywords[:5]),
            'temas': ', '.join(keywords)
        })
    
    # Guardar resultados
    city_folder = os.path.join(output_folder, city_name)
    os.makedirs(city_folder, exist_ok=True)
    
    # Archivo principal
    final_path = os.path.join(city_folder, f"analisis_{city_name}.csv")
    final_df.to_csv(final_path, index=False)
    
    # Resumen de clusters
    summary_path = os.path.join(city_folder, f"resumen_clusters_{city_name}.csv")
    pd.DataFrame(cluster_summary).to_csv(summary_path, index=False)
    
    print(f"✅ {city_name} procesado:")
    print(f"   - Archivo principal: {final_path}")
    print(f"   - Resumen clusters: {summary_path}")
    
    return final_df, cluster_summary

# Procesar todos los archivos en una carpeta
def process_cities(input_folder, output_folder='post_finales', n_clusters=6):
    # Crear directorio de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # Procesar cada archivo CSV
    all_summaries = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            _, summary = process_file(file_path, output_folder, n_clusters)
            
            if summary:
                all_summaries.extend(summary)
    
    # Guardar resumen global
    if all_summaries:
        global_summary = pd.DataFrame(all_summaries)
        global_path = os.path.join(output_folder, "resumen_global_ciudades.csv")
        global_summary.to_csv(global_path, index=False)
        print(f"\n🌎 Resumen global guardado: {global_path}")
    
    print("\n🚀 Proceso completo!")

# Ejecutar el procesamiento
if __name__ == '__main__':
    # Configuración
    input_folder = "/home/trigu3roslalo/Documentos/LIDIA - Universidad de Guanajuato/6° Semestre/XXX_Verano_de_la_Ciencia/verano/publicacion_corpus_limpio"  # Carpeta con CDMX.csv, Guadalajara.csv, Monterrey.csv
    n_clusters_por_ciudad = 5  # Puedes ajustar por ciudad
    
    # Ejecutar procesamiento
    process_cities(input_folder, n_clusters=n_clusters_por_ciudad)