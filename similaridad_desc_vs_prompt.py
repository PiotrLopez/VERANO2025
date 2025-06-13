import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """Limpia y preprocesa el texto"""
    if pd.isna(text):
        return ""
    # Convertir a minúsculas y eliminar caracteres especiales
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_corpus(texts):
    """Crea un corpus único concatenando todos los textos"""
    # Filtrar textos vacíos y unir en un solo corpus
    valid_texts = [text for text in texts if text.strip()]
    corpus = ' '.join(valid_texts)
    return corpus

def extract_ngrams_from_corpus(corpus, ngram_range=(1, 3), max_features=1000):
    """Extrae n-gramas de un corpus y obtiene los más frecuentes"""
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=None  # Puedes cambiar a 'english' o 'spanish' si quieres
    )
    
    try:
        # Ajustar el vectorizador al corpus
        vectorizer.fit([corpus])
        
        # Obtener la matriz de frecuencias
        freq_matrix = vectorizer.transform([corpus])
        
        # Obtener nombres de características (n-gramas) y sus frecuencias
        feature_names = vectorizer.get_feature_names_out()
        frequencies = freq_matrix.toarray()[0]
        
        # Crear diccionario de n-gramas con sus frecuencias
        ngram_freq = dict(zip(feature_names, frequencies))
        
        # Ordenar por frecuencia descendente
        sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_ngrams, vectorizer
        
    except Exception as e:
        print(f"    ❌ Error extrayendo n-gramas: {str(e)}")
        return [], None

def get_top_ngrams(sorted_ngrams, n_top):
    """Obtiene los n n-gramas más frecuentes"""
    return [ngram for ngram, freq in sorted_ngrams[:n_top]]

def create_corpus_vector(corpus, top_ngrams):
    """Crea un vector binario para un corpus basado en los n-gramas seleccionados"""
    # Crear vocabulario con los n-gramas seleccionados
    vocabulary = {ngram: i for i, ngram in enumerate(top_ngrams)}
    
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),  # Mismo rango que el análisis original
        vocabulary=vocabulary,
        binary=True  # Presencia/ausencia
    )
    
    try:
        vector = vectorizer.fit_transform([corpus])
        return vector.toarray()[0]
    except:
        # Si no hay coincidencias, crear vector de ceros
        return np.zeros(len(top_ngrams))

def jaccard_similarity(vec1, vec2):
    """Calcula la similitud de Jaccard entre dos vectores binarios"""
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    
    if union == 0:
        return 0.0
    return intersection / union

def jaccard_distance(vec1, vec2):
    """Calcula la distancia de Jaccard (1 - similitud)"""
    return 1 - jaccard_similarity(vec1, vec2)

def calculate_corpus_similarity(prompt_corpus, desc_corpus, n_words_list=[10, 20, 30, 40, 50]):
    """Calcula similitudes entre dos corpus usando n-gramas"""
    print("  🔤 Extrayendo n-gramas de corpus de prompts...")
    prompt_ngrams, prompt_vectorizer = extract_ngrams_from_corpus(prompt_corpus)
    
    print("  🔤 Extrayendo n-gramas de corpus de descripciones...")
    desc_ngrams, desc_vectorizer = extract_ngrams_from_corpus(desc_corpus)
    
    if not prompt_ngrams or not desc_ngrams:
        print("    ❌ No se pudieron extraer n-gramas de uno o ambos corpus")
        return {}
    
    print(f"    📊 N-gramas extraídos - Prompts: {len(prompt_ngrams)}, Descripciones: {len(desc_ngrams)}")
    
    # Combinar todos los n-gramas para crear vocabulario común
    all_ngrams = {}
    
    # Agregar n-gramas de prompts
    for ngram, freq in prompt_ngrams:
        all_ngrams[ngram] = all_ngrams.get(ngram, 0) + freq
    
    # Agregar n-gramas de descripciones
    for ngram, freq in desc_ngrams:
        all_ngrams[ngram] = all_ngrams.get(ngram, 0) + freq
    
    # Ordenar por frecuencia total
    sorted_combined_ngrams = sorted(all_ngrams.items(), key=lambda x: x[1], reverse=True)
    
    results = {}
    
    for n_words in n_words_list:
        print(f"  🔍 Analizando con top {n_words} n-gramas...")
        
        # Obtener los n n-gramas más frecuentes del vocabulario combinado
        top_ngrams = get_top_ngrams(sorted_combined_ngrams, n_words)
        
        if len(top_ngrams) == 0:
            print(f"    ❌ No hay n-gramas suficientes para análisis con {n_words}")
            continue
        
        # Crear vectores para cada corpus
        prompt_vector = create_corpus_vector(prompt_corpus, top_ngrams)
        desc_vector = create_corpus_vector(desc_corpus, top_ngrams)
        
        # Calcular similitudes
        jac_distance = jaccard_distance(prompt_vector, desc_vector)
        jac_similarity = 1 - jac_distance
        
        # Para similitud coseno, necesitamos vectores 2D
        cos_similarity = cosine_similarity([prompt_vector], [desc_vector])[0][0]
        
        # Estadísticas adicionales
        prompt_coverage = np.sum(prompt_vector) / len(top_ngrams)
        desc_coverage = np.sum(desc_vector) / len(top_ngrams)
        intersection_count = np.sum(np.logical_and(prompt_vector, desc_vector))
        
        results[f'corpus_comparison_{n_words}'] = {
            'top_ngrams_count': n_words,
            'actual_ngrams_used': len(top_ngrams),
            'jaccard_distance': jac_distance,
            'jaccard_similarity': jac_similarity,
            'cosine_similarity': cos_similarity,
            'prompt_coverage': prompt_coverage,  # % de n-gramas presentes en prompts
            'desc_coverage': desc_coverage,      # % de n-gramas presentes en descripciones
            'intersection_count': intersection_count,  # N-gramas comunes
            'top_ngrams': top_ngrams[:10],  # Los 10 n-gramas más frecuentes
            'prompt_unique_ngrams': len([ngram for ngram, _ in prompt_ngrams]),
            'desc_unique_ngrams': len([ngram for ngram, _ in desc_ngrams])
        }
        
        print(f"    ✅ Similitud Jaccard: {jac_similarity:.4f}, Coseno: {cos_similarity:.4f}")
    
    return results

def process_csv_file(file_path, n_words_list=[10, 20, 30, 40, 50]):
    """Procesa un archivo CSV y calcula similitudes entre corpus"""
    print(f"\nProcesando: {os.path.basename(file_path)}")
    
    try:
        # Leer CSV
        df = pd.read_csv(file_path)
        
        # Verificar que las columnas existan
        required_columns = ["prompt_procesado_es", "descripcion_procesada_es"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  ❌ Columnas faltantes: {missing_columns}")
            return None
        
        # Limpiar textos
        df['prompt_clean'] = df['prompt_procesado_es'].apply(clean_text)
        df['descripcion_clean'] = df['descripcion_procesada_es'].apply(clean_text)
        
        # Crear corpus separados
        prompt_texts = df['prompt_clean'].tolist()
        desc_texts = df['descripcion_clean'].tolist()
        
        prompt_corpus = create_corpus(prompt_texts)
        desc_corpus = create_corpus(desc_texts)
        
        if not prompt_corpus.strip() or not desc_corpus.strip():
            print(f"  ❌ Uno o ambos corpus están vacíos")
            return None
        
        print(f"  📊 Corpus creados:")
        print(f"    - Corpus prompts: {len(prompt_corpus.split()):,} palabras")
        print(f"    - Corpus descripciones: {len(desc_corpus.split()):,} palabras")
        
        # Calcular similitudes entre corpus
        results = calculate_corpus_similarity(prompt_corpus, desc_corpus, n_words_list)
        
        return {
            'file': os.path.basename(file_path),
            'n_rows': len(df),
            'prompt_corpus_words': len(prompt_corpus.split()),
            'desc_corpus_words': len(desc_corpus.split()),
            'results': results
        }
        
    except Exception as e:
        print(f"  ❌ Error procesando archivo: {str(e)}")
        return None

def save_results_to_csv(all_results, output_file='corpus_ngram_similarity_aca_can.csv'):
    """Guarda los resultados en un archivo CSV"""
    rows = []
    
    for file_result in all_results:
        if file_result is None:
            continue
            
        file_name = file_result['file']
        n_rows = file_result['n_rows']
        prompt_words = file_result['prompt_corpus_words']
        desc_words = file_result['desc_corpus_words']
        results = file_result['results']
        
        for analysis_key, metrics in results.items():
            rows.append({
                'archivo': file_name,
                'filas_totales': n_rows,
                'palabras_corpus_prompts': prompt_words,
                'palabras_corpus_descripciones': desc_words,
                'top_ngrams_solicitados': metrics['top_ngrams_count'],
                'ngrams_utilizados': metrics['actual_ngrams_used'],
                'jaccard_distance': round(metrics['jaccard_distance'], 6),
                'jaccard_similarity': round(metrics['jaccard_similarity'], 6),
                'cosine_similarity': round(metrics['cosine_similarity'], 6),
                'coverage_prompts': round(metrics['prompt_coverage'], 4),
                'coverage_descripciones': round(metrics['desc_coverage'], 4),
                'ngrams_comunes': metrics['intersection_count'],
                'ngrams_unicos_prompts': metrics['prompt_unique_ngrams'],
                'ngrams_unicos_descripciones': metrics['desc_unique_ngrams'],
                'top_10_ngrams': ' | '.join(metrics['top_ngrams'])
            })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 Resultados guardados en: {output_file}")
    
    return df_results

def print_summary(df_results):
    """Imprime un resumen de los resultados"""
    print("\n📈 RESUMEN DE SIMILITUDES ENTRE CORPUS:")
    print("=" * 80)
    
    for _, row in df_results.iterrows():
        print(f"\n📁 {row['archivo']} - Top {row['top_ngrams_solicitados']} n-gramas")
        print(f"   Corpus - Prompts: {row['palabras_corpus_prompts']:,} palabras | Descripciones: {row['palabras_corpus_descripciones']:,} palabras")
        print(f"   Distancia Jaccard: {row['jaccard_distance']:.6f}")
        print(f"   Similitud Jaccard: {row['jaccard_similarity']:.6f}")
        print(f"   Similitud Coseno:  {row['cosine_similarity']:.6f}")
        print(f"   N-gramas comunes:  {row['ngrams_comunes']}/{row['ngrams_utilizados']}")
        print(f"   Cobertura: Prompts {row['coverage_prompts']:.1%} | Descripciones {row['coverage_descripciones']:.1%}")

def main():
    """Función principal"""
    # Configurar carpeta de archivos CSV
    folder_path = input("Ingresa la ruta de la carpeta con los archivos CSV (o presiona Enter para usar la carpeta actual): ").strip()
    
    if not folder_path:
        folder_path = "."
    
    # Buscar archivos CSV
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"❌ No se encontraron archivos CSV en: {folder_path}")
        return
    
    print(f"\n🔍 Encontrados {len(csv_files)} archivos CSV:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Procesar archivos
    print(f"\n🚀 Iniciando análisis de similitud entre corpus usando n-gramas...")
    print("🎯 Comparando CORPUS de prompts vs CORPUS de descripciones")
    print("📝 Usando n-gramas (unigramas, bigramas, trigramas)")
    
    all_results = []
    
    for csv_file in csv_files:
        result = process_csv_file(csv_file)
        all_results.append(result)
    
    # Guardar y mostrar resultados
    if any(result is not None for result in all_results):
        df_results = save_results_to_csv(all_results)
        print_summary(df_results)
        
        valid_results = [r for r in all_results if r is not None]
        
        print(f"\n✅ Análisis de corpus completado!")
        print(f"📁 Archivos procesados: {len(valid_results)}")
        print(f"🔤 N-gramas analizados: unigramas, bigramas y trigramas")
        print(f"📊 Palabras analizadas: 10, 20, 30, 40, 50 más frecuentes")
    else:
        print("\n❌ No se pudieron procesar archivos.")

if __name__ == "__main__":
    main()