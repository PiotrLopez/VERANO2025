# import pandas as pd
# import numpy as np
# import os
# import networkx as nx
# from pyvis.network import Network
# from collections import Counter
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from textblob import TextBlob
# from textblob_fr import PatternTagger, PatternAnalyzer  # Para análisis en español

# # --------------------------------
# # Configuración inicial
# # --------------------------------
# print("🔍 Inicializando sistema de análisis comparativo...")

# POST_FINALES_PATH = "post_finales"
# DESCRIPCIONES_FINALES_PATH = "descripciones_finales"
# ANALISIS_PATH = "analisis"

# os.makedirs(ANALISIS_PATH, exist_ok=True)

# print("⏳ Cargando modelo de embeddings ...")
# sentence_model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")

# print("⏳ Cargando modelo de spaCy...")
# nlp = spacy.load("es_core_news_sm")

# # -----------------------------------
# # Funciones auxiliares
# # -----------------------------------

# def analyze_sentiment(text):
#     """Análisis de sentimiento usando TextBlob para español"""
#     try:
#         blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
#         polarity = blob.sentiment[0]
#         if polarity > 0.2:
#             return "POS"
#         elif polarity < -0.2:
#             return "NEG"
#         else:
#             return "NEU"
#     except:
#         return "NEU"

# def detect_frames(text):
#     """Detección de marcos conceptuales usando reglas léxicas"""
#     text = text.lower()
#     frames = []
#     problem_keywords = ["problema", "desafío", "dificultad", "retro", "obstáculo"]
#     solution_keywords = ["solución", "resolver", "mejorar", "propuesta", "plan"]
#     if any(kw in text for kw in problem_keywords) and any(kw in text for kw in solution_keywords):
#         frames.append("PROBLEMA_SOLUCION")
#     conflict_keywords = ["conflicto", "disputa", "debate", "controversia", "oposición"]
#     if any(kw in text for kw in conflict_keywords):
#         frames.append("CONFLICTO")
#     opportunity_keywords = ["oportunidad", "beneficio", "ventaja", "potencial", "crecimiento"]
#     if any(kw in text for kw in opportunity_keywords):
#         frames.append("OPORTUNIDAD")
#     responsibility_keywords = ["responsabilidad", "deber", "obligación", "compromiso", "cuenta"]
#     if any(kw in text for kw in responsibility_keywords):
#         frames.append("RESPONSABILIDAD")
#     return ", ".join(frames) if frames else "NEUTRAL"

# def bertopic_subtopics(df):
#     """Aplica BERTopic para descubrir subtemas dentro de clusters"""
#     print("🔍 Aplicando BERTopic para subtemas...")
#     if 'cluster' not in df.columns:
#         print("⚠️ No se encontró columna 'cluster'. Saltando BERTopic.")
#         return df
#     df['subtopico'] = -1
#     for cluster_id in tqdm(df['cluster'].unique(), desc="Modelando subtemas por cluster"):
#         cluster_mask = df['cluster'] == cluster_id
#         cluster_texts = df[cluster_mask]['texto_procesado'].tolist()
#         if len(cluster_texts) > 5:
#             topic_model = BERTopic(
#                 embedding_model=sentence_model,
#                 language="spanish",
#                 nr_topics="auto",
#                 verbose=False
#             )
#             try:
#                 topics, _ = topic_model.fit_transform(cluster_texts)
#                 df.loc[cluster_mask, 'subtopico'] = topics
#             except Exception as e:
#                 print(f"⚠️ Error en cluster {cluster_id}: {str(e)}")
#     return df

# def compute_cluster_statistics(df):
#     """Genera resumen cuantitativo por cluster con distribución de posts y descripciones, sentimientos y frames."""
#     stats = []
#     clusters = sorted(df['cluster'].unique())
#     for cl in clusters:
#         df_cl = df[df['cluster'] == cl]
#         n_total = len(df_cl)
#         n_post = sum(df_cl['fuente'] == 'post')
#         n_desc = sum(df_cl['fuente'] == 'descripcion')

#         sentiment_counts = df_cl['sentimiento'].value_counts(normalize=True).to_dict()
#         frame_counts = df_cl['frame'].value_counts(normalize=True).to_dict()

#         stats.append({
#             "cluster": cl,
#             "total_docs": n_total,
#             "posts_count": n_post,
#             "desc_count": n_desc,
#             "sentiment_dist": sentiment_counts,
#             "frame_dist": frame_counts
#         })
#     return stats

# def calculate_inter_set_similarity(df_post, df_desc):
#     """Calcula la similitud promedio entre textos procesados de posts y descripciones por cluster"""
#     cluster_ids = set(df_post['cluster'].unique()).intersection(df_desc['cluster'].unique())
#     similarity_per_cluster = {}
#     for cl in cluster_ids:
#         texts_post = df_post[df_post['cluster'] == cl]['texto_procesado'].tolist()
#         texts_desc = df_desc[df_desc['cluster'] == cl]['texto_procesado'].tolist()
        
#         # Compute average embeddings
#         emb_post = sentence_model.encode(texts_post)
#         emb_desc = sentence_model.encode(texts_desc)
        
#         # Matriz de similitud cosine entre los dos sets
#         sim_matrix = cosine_similarity(emb_post, emb_desc)
#         avg_sim = np.mean(sim_matrix)
#         similarity_per_cluster[cl] = avg_sim
#     return similarity_per_cluster

# def build_comparative_network(df_post, df_desc):
#     """Construye red de clusters diferenciando fuente, con enlaces que representan similitud semántica inter-conjunto"""
#     G = nx.Graph()

#     # Añadir nodos para posts
#     for cl in df_post['cluster'].unique():
#         cluster_df = df_post[df_post['cluster'] == cl]
#         vectorizer = TfidfVectorizer(max_features=5)
#         tfidf_matrix = vectorizer.fit_transform(cluster_df['texto_procesado'])
#         feature_array = vectorizer.get_feature_names_out()
#         keywords = ", ".join(feature_array)
#         sentiments = cluster_df['sentimiento'].value_counts()
#         dominant_sent = sentiments.idxmax() if not sentiments.empty else "NEU"
#         frames = [f for f in cluster_df['frame'] if f != "NEUTRAL"]
#         dominant_frame = Counter(frames).most_common(1)[0][0] if frames else "NEUTRAL"

#         G.add_node(f"post_cluster_{cl}",
#                    label=f"Post Cluster {cl}",
#                    size=len(cluster_df),
#                    fuente="post",
#                    keywords=keywords,
#                    sentimiento=dominant_sent,
#                    frame=dominant_frame)

#     # Añadir nodos para descripciones
#     for cl in df_desc['cluster'].unique():
#         cluster_df = df_desc[df_desc['cluster'] == cl]
#         vectorizer = TfidfVectorizer(max_features=5)
#         tfidf_matrix = vectorizer.fit_transform(cluster_df['texto_procesado'])
#         feature_array = vectorizer.get_feature_names_out()
#         keywords = ", ".join(feature_array)
#         sentiments = cluster_df['sentimiento'].value_counts()
#         dominant_sent = sentiments.idxmax() if not sentiments.empty else "NEU"
#         frames = [f for f in cluster_df['frame'] if f != "NEUTRAL"]
#         dominant_frame = Counter(frames).most_common(1)[0][0] if frames else "NEUTRAL"

#         G.add_node(f"desc_cluster_{cl}",
#                    label=f"Desc Cluster {cl}",
#                    size=len(cluster_df),
#                    fuente="descripcion",
#                    keywords=keywords,
#                    sentimiento=dominant_sent,
#                    frame=dominant_frame)

#     # Conectar clusters post y descripción por similitud semántica mayor a un umbral
#     common_clusters = set(df_post['cluster'].unique()).intersection(df_desc['cluster'].unique())
#     for cl in common_clusters:
#         texts_post = df_post[df_post['cluster'] == cl]['texto_procesado'].tolist()
#         texts_desc = df_desc[df_desc['cluster'] == cl]['texto_procesado'].tolist()

#         emb_post = sentence_model.encode(texts_post)
#         emb_desc = sentence_model.encode(texts_desc)

#         sim_matrix = cosine_similarity(emb_post, emb_desc)
#         mean_sim = np.mean(sim_matrix)
#         # Consideramos añadir enlace si similitud media supera 0.6 (se puede ajustar)
#         if mean_sim > 0.6:
#             G.add_edge(f"post_cluster_{cl}", f"desc_cluster_{cl}", weight=mean_sim, label=f"Simil: {mean_sim:.2f}")

#     return G

# def generate_comparative_report(df_post, df_desc, city_name):
#     print(f"📝 Generando reporte comparativo para {city_name}...")

#     stats_post = compute_cluster_statistics(df_post)
#     stats_desc = compute_cluster_statistics(df_desc)
#     similarity_per_cluster = calculate_inter_set_similarity(df_post, df_desc)

#     report_lines = []
#     report_lines.append(f"# Reporte Comparativo de Posts y Descripciones - {city_name}\n")
#     report_lines.append(f"## Resumen General\n")
#     report_lines.append(f"- Total Posts: {len(df_post)}")
#     report_lines.append(f"- Total Descripciones: {len(df_desc)}\n")

#     report_lines.append("## Análisis por Cluster\n")
#     all_clusters = sorted(set(df_post['cluster'].unique()).union(set(df_desc['cluster'].unique())))

#     for cl in all_clusters:
#         report_lines.append(f"### Cluster {cl}")
#         # Estadísticas posts
#         stat_p = next((item for item in stats_post if item["cluster"] == cl), None)
#         stat_d = next((item for item in stats_desc if item["cluster"] == cl), None)

#         report_lines.append(f"- Posts: {stat_p['total_docs'] if stat_p else 0}")
#         report_lines.append(f"- Descripciones: {stat_d['total_docs'] if stat_d else 0}")

#         # Sentimientos
#         sent_p = stat_p['sentiment_dist'] if stat_p else {}
#         sent_d = stat_d['sentiment_dist'] if stat_d else {}
#         report_lines.append("- Sentimientos Posts: " + ", ".join([f"{k}: {v:.1%}" for k,v in sent_p.items()]))
#         report_lines.append("- Sentimientos Descripciones: " + ", ".join([f"{k}: {v:.1%}" for k,v in sent_d.items()]))

#         # Marcos
#         frame_p = stat_p['frame_dist'] if stat_p else {}
#         frame_d = stat_d['frame_dist'] if stat_d else {}
#         report_lines.append("- Marcos Posts: " + ", ".join([f"{k}: {v:.1%}" for k,v in frame_p.items()]))
#         report_lines.append("- Marcos Descripciones: " + ", ".join([f"{k}: {v:.1%}" for k,v in frame_d.items()]))

#         # Similitud semántica promedio
#         sim = similarity_per_cluster.get(cl, None)
#         if sim is not None:
#             report_lines.append(f"- Similitud semántica promedio entre Posts y Descripciones: {sim:.4f}")
#         else:
#             report_lines.append(f"- Similitud semántica promedio entre Posts y Descripciones: N/A")

#         report_lines.append("")

#     # Guardar reporte
#     report_dir = os.path.join(ANALISIS_PATH, city_name)
#     os.makedirs(report_dir, exist_ok=True)
#     report_path = os.path.join(report_dir, f"reporte_comparativo_{city_name}.md")
#     with open(report_path, "w", encoding='utf-8') as f:
#         f.write("\n".join(report_lines))

#     print(f"✅ Reporte comparativo guardado en: {report_path}")
#     return report_path

# def visualize_network(G, city_name):
#     """Crea visualización interactiva de la red de clusters"""
#     if not G:
#         return None
#     print("🎨 Generando visualización de red...")
#     net = Network(height="800px", width="100%", notebook=False)
#     net.from_nx(G)
#     sentiment_colors = {
#         'POS': '#4CAF50',
#         'NEG': '#F44336',
#         'NEU': '#9E9E9E'
#     }
#     frame_shapes = {
#         'PROBLEMA_SOLUCION': 'triangle',
#         'CONFLICTO': 'square',
#         'OPORTUNIDAD': 'star',
#         'RESPONSABILIDAD': 'diamond',
#         'NEUTRAL': 'circle'
#     }
#     for node in net.nodes:
#         sentiment = node.get('sentimiento', 'NEU')
#         frame = node.get('frame', 'NEUTRAL')
#         node['color'] = sentiment_colors.get(sentiment, '#9E9E9E')
#         node['shape'] = frame_shapes.get(frame, 'circle')
#         node_info = [
#             f"<b>{node['label']}</b>",
#             f"Palabras clave: {node.get('keywords', '')}",
#             f"Sentimiento: {sentiment}",
#             f"Marco: {frame}",
#             f"Tamaño: {node.get('size', 0)}"
#         ]
#         node['title'] = "<br>".join(node_info)
#     net.barnes_hut(
#         gravity=-80000,
#         central_gravity=0.3,
#         spring_length=250,
#         spring_strength=0.001,
#         damping=0.09,
#         overlap=0
#     )
#     net_path = os.path.join(ANALISIS_PATH, city_name, f"red_tematica_{city_name}.html")
#     net.save_graph(net_path)
#     print(f"✅ Visualización de red guardada: {net_path}")
#     return net_path

# def process_city(city_name):
#     print(f"\n{'='*50}")
#     print(f"🏙️ PROCESANDO CIUDAD: {city_name}")
#     print(f"{'='*50}")

#     post_folder = f"{city_name}_CORPUS_LIMPIO"
#     desc_folder = f"{city_name}_DESC_TRADUCIDO_LIMPIO"

#     path_post = os.path.join(POST_FINALES_PATH, post_folder, f"analisis_{post_folder}.csv")
#     path_desc = os.path.join(DESCRIPCIONES_FINALES_PATH, desc_folder, f"analisis_{desc_folder}.csv")

#     print(f"Buscando posts en: {path_post}")
#     print(f"Buscando descripciones en: {path_desc}")

#     if not (os.path.exists(path_post) and os.path.exists(path_desc)):
#         print(f"⚠️ Archivo(s) faltante(s) para {city_name}, saltando análisis.")
#         return

#     df_post = pd.read_csv(path_post)
#     df_desc = pd.read_csv(path_desc)

#     df_post['fuente'] = 'post'
#     df_desc['fuente'] = 'descripcion'

#     # Aplicar análisis individuales (sentimiento, frames)
#     tqdm.pandas(desc="Analizando sentimiento en posts")
#     df_post['sentimiento'] = df_post['texto_procesado'].progress_apply(analyze_sentiment)
#     tqdm.pandas(desc="Analizando sentimiento en descripciones")
#     df_desc['sentimiento'] = df_desc['texto_procesado'].progress_apply(analyze_sentiment)

#     tqdm.pandas(desc="Detectando marcos en posts")
#     df_post['frame'] = df_post['texto_procesado'].progress_apply(detect_frames)
#     tqdm.pandas(desc="Detectando marcos en descripciones")
#     df_desc['frame'] = df_desc['texto_procesado'].progress_apply(detect_frames)

#     # Modelado de subtemas para cada conjunto
#     df_post = bertopic_subtopics(df_post)
#     df_desc = bertopic_subtopics(df_desc)

#     # Construcción de red integrada
#     G = build_comparative_network(df_post, df_desc)

#     # Generar reporte comparativo
#     report_path = generate_comparative_report(df_post, df_desc, city_name)

#     # Visualización de red
#     net_path = visualize_network(G, city_name)

#     # Guardar datasets analizados separados y combinados
#     output_dir = os.path.join(ANALISIS_PATH, city_name)
#     os.makedirs(output_dir, exist_ok=True)
#     df_post.to_csv(os.path.join(output_dir, f"post_analizado_{city_name}.csv"), index=False)
#     df_desc.to_csv(os.path.join(output_dir, f"desc_analizado_{city_name}.csv"), index=False)
#     df_combined = pd.concat([df_post, df_desc], ignore_index=True)
#     df_combined.to_csv(os.path.join(output_dir, f"combinado_analizado_{city_name}.csv"), index=False)

#     print(f"💾 Datos analizados guardados en: {output_dir}")

#     return report_path, net_path

# if __name__ == "__main__":
#     cities = []
#     if os.path.exists(POST_FINALES_PATH):
#         cities = list(set([d.split('_')[0] for d in os.listdir(POST_FINALES_PATH) if os.path.isdir(os.path.join(POST_FINALES_PATH, d))]))
#     print("\nCiudades detectadas para análisis:")
#     for idx, c in enumerate(sorted(cities), 1):
#         print(f"{idx}. {c}")

#     for city in sorted(cities):
#         process_city(city)

#     print("\n🚀 Análisis comparativo completado!")



import pandas as pd
import numpy as np
import os
import networkx as nx
from pyvis.network import Network
from collections import Counter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Configuración inicial
# --------------------------------
print("🔍 Inicializando sistema de análisis comparativo...")

POST_FINALES_PATH = "post_finales"
DESCRIPCIONES_FINALES_PATH = "descripciones_finales"
ANALISIS_PATH = "analisis"

os.makedirs(ANALISIS_PATH, exist_ok=True)

print("⏳ Cargando modelo de embeddings ...")
sentence_model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")

print("⏳ Cargando modelo de spaCy...")
nlp = spacy.load("es_core_news_sm")

# -----------------------------------
# Funciones auxiliares
# -----------------------------------

def analyze_sentiment(text):
    """Análisis de sentimiento usando TextBlob para español"""
    try:
        blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        polarity = blob.sentiment[0]
        if polarity > 0.2:
            return "POS"
        elif polarity < -0.2:
            return "NEG"
        else:
            return "NEU"
    except:
        return "NEU"

def detect_frames(text):
    """Detección de marcos conceptuales usando reglas léxicas"""
    text = text.lower()
    frames = []
    problem_keywords = ["problema", "desafío", "dificultad", "retro", "obstáculo", "falla"]
    solution_keywords = ["solución", "resolver", "mejorar", "propuesta", "plan", "remedio"]
    if any(kw in text for kw in problem_keywords) and any(kw in text for kw in solution_keywords):
        frames.append("PROBLEMA_SOLUCION")
    conflict_keywords = ["conflicto", "disputa", "debate", "controversia", "oposición", "pelea"]
    if any(kw in text for kw in conflict_keywords):
        frames.append("CONFLICTO")
    opportunity_keywords = ["oportunidad", "beneficio", "ventaja", "potencial", "crecimiento", "opción"]
    if any(kw in text for kw in opportunity_keywords):
        frames.append("OPORTUNIDAD")
    responsibility_keywords = ["responsabilidad", "deber", "obligación", "compromiso", "cuenta", "incumbencia"]
    if any(kw in text for kw in responsibility_keywords):
        frames.append("RESPONSABILIDAD")
    return ", ".join(frames) if frames else "NEUTRAL"

def bertopic_subtopics(df):
    """Aplica BERTopic para descubrir subtemas dentro de clusters"""
    print("🔍 Aplicando BERTopic para subtemas...")
    if 'cluster' not in df.columns:
        print("⚠️ No se encontró columna 'cluster'. Saltando BERTopic.")
        return df
    df['subtopico'] = -1
    for cluster_id in tqdm(df['cluster'].unique(), desc="Modelando subtemas por cluster"):
        cluster_mask = df['cluster'] == cluster_id
        cluster_texts = df[cluster_mask]['texto_procesado'].tolist()
        if len(cluster_texts) > 5:
            topic_model = BERTopic(
                embedding_model=sentence_model,
                language="spanish",
                nr_topics="auto",
                verbose=False
            )
            try:
                topics, _ = topic_model.fit_transform(cluster_texts)
                df.loc[cluster_mask, 'subtopico'] = topics
            except Exception as e:
                print(f"⚠️ Error en cluster {cluster_id}: {str(e)}")
    return df

def extract_cluster_keywords(df, cluster_col='cluster', text_col='texto_procesado', top_n=10):
    """Extrae palabras clave por cluster usando TF-IDF"""
    cluster_keywords = {}
    for cluster_id in df[cluster_col].unique():
        cluster_texts = df[df[cluster_col] == cluster_id][text_col].tolist()
        if cluster_texts:
            vectorizer = TfidfVectorizer(max_features=top_n, stop_words=list(spacy.lang.es.stop_words.STOP_WORDS))
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_array = vectorizer.get_feature_names_out()
                keywords = ", ".join(feature_array)
            except ValueError:
                keywords = "Sin palabras clave suficientes"
        else:
            keywords = "Sin textos"
        cluster_keywords[cluster_id] = keywords
    return cluster_keywords

def generate_advanced_report(df, city_name, source_type):
    """Genera reporte avanzado en el formato solicitado"""
    print(f"📝 Generando reporte avanzado para {source_type} de {city_name}...")
    
    # Estadísticas generales
    total_docs = len(df)
    n_clusters = df['cluster'].nunique()
    
    # Distribución de sentimientos
    sentiment_counts = df['sentimiento'].value_counts()
    sentiment_dist = {k: v for k, v in sentiment_counts.items()}
    
    # Distribución de marcos
    frame_counts = df['frame'].value_counts()
    frame_dist = {k: v for k, v in frame_counts.items()}
    
    # Palabras clave por cluster
    cluster_keywords = extract_cluster_keywords(df)
    
    # Sentimiento y marco predominante por cluster
    cluster_stats = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        # Sentimiento predominante
        sentiment_counts = cluster_df['sentimiento'].value_counts()
        dominant_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else "NEU"
        
        # Marco predominante
        frame_counts = cluster_df['frame'].value_counts()
        dominant_frame = frame_counts.idxmax() if not frame_counts.empty else "NEUTRAL"
        
        cluster_stats[cluster_id] = {
            'size': cluster_size,
            'keywords': cluster_keywords[cluster_id],
            'sentiment': dominant_sentiment,
            'frame': dominant_frame
        }
    
    # Calcular similitudes entre clusters
    cluster_embeddings = {}
    for cluster_id in df['cluster'].unique():
        cluster_texts = df[df['cluster'] == cluster_id]['texto_procesado'].tolist()
        if cluster_texts:
            embeddings = sentence_model.encode(cluster_texts)
            avg_embedding = np.mean(embeddings, axis=0)
            cluster_embeddings[cluster_id] = avg_embedding
    
    # Matriz de similitud
    cluster_ids = sorted(cluster_embeddings.keys())
    similarity_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))
    for i, cl_i in enumerate(cluster_ids):
        for j, cl_j in enumerate(cluster_ids):
            if i < j:  # Evitar duplicados y diagonal
                sim = cosine_similarity([cluster_embeddings[cl_i]], [cluster_embeddings[cl_j]])[0][0]
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # Construir reporte
    report_lines = []
    report_lines.append(f"# Análisis Avanzado - {city_name} - {source_type.capitalize()}\n")
    report_lines.append(f"## Resumen General\n")
    report_lines.append(f"- Documentos analizados: {total_docs}")
    report_lines.append(f"- Clusters identificados: {n_clusters}")
    
    report_lines.append("- Distribución de sentimientos:")
    for sent, count in sentiment_dist.items():
        report_lines.append(f"  - {sent}: {count} ({count/total_docs:.1%})")
    
    report_lines.append("\n- Distribución de marcos:")
    for frame, count in frame_dist.items():
        report_lines.append(f"  - {frame}: {count} ({count/total_docs:.1%})")
    
    report_lines.append("\n## Análisis por Cluster\n")
    for cluster_id, stats in cluster_stats.items():
        report_lines.append(f"### Cluster {cluster_id} (Tamaño: {stats['size']})")
        report_lines.append(f"- **Palabras clave**: {stats['keywords']}")
        report_lines.append(f"- **Sentimiento predominante**: {stats['sentiment']}")
        report_lines.append(f"- **Marco predominante**: {stats['frame']}\n")
    
    # Sección de relaciones entre clusters
    report_lines.append("## Relaciones entre Clusters\n")
    
    # Obtener las conexiones más significativas
    significant_connections = []
    for i, cl_i in enumerate(cluster_ids):
        for j, cl_j in enumerate(cluster_ids):
            if i < j and similarity_matrix[i, j] > 0.7:  # Umbral de similitud
                significant_connections.append((cl_i, cl_j, similarity_matrix[i, j]))
    
    # Ordenar por similitud descendente
    significant_connections.sort(key=lambda x: x[2], reverse=True)
    
    if significant_connections:
        report_lines.append("### Conexiones Significativas")
        for cl_i, cl_j, sim in significant_connections:
            report_lines.append(f"- **Cluster_{cl_i} ↔ Cluster_{cl_j}**: Similitud = {sim:.2f}")
    else:
        report_lines.append("### No se encontraron conexiones significativas (similitud > 0.7)")
    
    # Guardar reporte
    report_dir = os.path.join(ANALISIS_PATH, city_name)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"reporte_avanzado_{source_type}_{city_name}.md")
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    # Generar visualización de la matriz de similitud
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        xticklabels=cluster_ids,
        yticklabels=cluster_ids
    )
    plt.title(f"Matriz de Similitud entre Clusters - {city_name} - {source_type}")
    plt.tight_layout()
    heatmap_path = os.path.join(report_dir, f"heatmap_similitud_{source_type}_{city_name}.png")
    plt.savefig(heatmap_path)
    plt.close()
    
    print(f"✅ Reporte avanzado guardado: {report_path}")
    print(f"✅ Heatmap de similitud guardado: {heatmap_path}")
    
    return report_path, heatmap_path

def build_comparative_network(df_post, df_desc, city_name):
    """Construye red de clusters diferenciando fuente, con enlaces que representan similitud semántica"""
    G = nx.Graph()
    report_dir = os.path.join(ANALISIS_PATH, city_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # Color y forma por fuente y características
    source_colors = {
        'post': '#1f78b4',    # Azul para posts
        'descripcion': '#33a02c'  # Verde para descripciones
    }
    
    sentiment_shapes = {
        'POS': 'triangle',
        'NEG': 'square',
        'NEU': 'circle'
    }
    
    # Añadir nodos para posts
    for cl in df_post['cluster'].unique():
        cluster_df = df_post[df_post['cluster'] == cl]
        keywords = extract_cluster_keywords(cluster_df).get(cl, "Sin palabras clave")
        
        sentiments = cluster_df['sentimiento'].value_counts()
        dominant_sent = sentiments.idxmax() if not sentiments.empty else "NEU"
        
        frames = cluster_df['frame'].value_counts()
        dominant_frame = frames.idxmax() if not frames.empty else "NEUTRAL"
        
        G.add_node(f"post_{cl}",
                   label=f"Post Cluster {cl}",
                   size=len(cluster_df),
                   fuente="post",
                   keywords=keywords,
                   sentimiento=dominant_sent,
                   frame=dominant_frame,
                   color=source_colors['post'],
                   shape=sentiment_shapes.get(dominant_sent, 'circle'))

    # Añadir nodos para descripciones
    for cl in df_desc['cluster'].unique():
        cluster_df = df_desc[df_desc['cluster'] == cl]
        keywords = extract_cluster_keywords(cluster_df).get(cl, "Sin palabras clave")
        
        sentiments = cluster_df['sentimiento'].value_counts()
        dominant_sent = sentiments.idxmax() if not sentiments.empty else "NEU"
        
        frames = cluster_df['frame'].value_counts()
        dominant_frame = frames.idxmax() if not frames.empty else "NEUTRAL"
        
        G.add_node(f"desc_{cl}",
                   label=f"Desc Cluster {cl}",
                   size=len(cluster_df),
                   fuente="descripcion",
                   keywords=keywords,
                   sentimiento=dominant_sent,
                   frame=dominant_frame,
                   color=source_colors['descripcion'],
                   shape=sentiment_shapes.get(dominant_sent, 'circle'))

    # Conectar clusters post y descripción por similitud semántica
    for post_cl in df_post['cluster'].unique():
        for desc_cl in df_desc['cluster'].unique():
            # Calcular similitud entre clusters
            post_texts = df_post[df_post['cluster'] == post_cl]['texto_procesado'].tolist()
            desc_texts = df_desc[df_desc['cluster'] == desc_cl]['texto_procesado'].tolist()
            
            if post_texts and desc_texts:
                emb_post = sentence_model.encode(post_texts)
                emb_desc = sentence_model.encode(desc_texts)
                
                sim_matrix = cosine_similarity(emb_post, emb_desc)
                mean_sim = np.mean(sim_matrix)
                
                # Añadir enlace si supera umbral y hay diferencia significativa
                if mean_sim > 0.5 and abs(post_cl - desc_cl) > 0.5:  # Umbral ajustable
                    G.add_edge(f"post_{post_cl}", f"desc_{desc_cl}", 
                              weight=mean_sim, 
                              label=f"Simil: {mean_sim:.2f}",
                              color="#e31a1c" if mean_sim < 0.7 else "#ff7f00")  # Rojo para baja similitud, naranja para media

    # Visualización interactiva con PyVis
    if G.nodes:
        net = Network(height="800px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
        net.from_nx(G)
        
        # Configurar física para mejor distribución
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.001,
            damping=0.09,
            overlap=0
        )
        
        # Personalizar tooltips
        for node in net.nodes:
            node_info = [
                f"<b>{node['label']}</b>",
                f"Tamaño: {node['size']}",
                f"Palabras clave: {node.get('keywords', 'N/A')}",
                f"Sentimiento: {node.get('sentimiento', 'N/A')}",
                f"Marco: {node.get('frame', 'N/A')}"
            ]
            node['title'] = "<br>".join(node_info)
        
        # Guardar red
        net_path = os.path.join(report_dir, f"red_comparativa_{city_name}.html")
        net.save_graph(net_path)
        print(f"✅ Visualización de red guardada: {net_path}")
        
        # Generar visualización estática para diferencias
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Dibujar nodos
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[n for n in G.nodes if 'post' in n],
            node_color=[G.nodes[n]['color'] for n in G.nodes if 'post' in n],
            node_size=[G.nodes[n]['size']*10 for n in G.nodes if 'post' in n],
            alpha=0.8,
            label="Posts"
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[n for n in G.nodes if 'desc' in n],
            node_color=[G.nodes[n]['color'] for n in G.nodes if 'desc' in n],
            node_size=[G.nodes[n]['size']*10 for n in G.nodes if 'desc' in n],
            alpha=0.8,
            label="Descripciones"
        )
        
        # Dibujar bordes
        edge_colors = [G.edges[e]['color'] for e in G.edges]
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=[G.edges[e]['weight']*5 for e in G.edges],
            alpha=0.6
        )
        
        # Dibujar etiquetas
        node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Dibujar etiquetas de bordes
        edge_labels = {(u, v): f"{G.edges[(u, v)]['weight']:.2f}" for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Red Comparativa: Posts vs Descripciones - {city_name}")
        plt.legend()
        plt.tight_layout()
        static_net_path = os.path.join(report_dir, f"red_comparativa_{city_name}.png")
        plt.savefig(static_net_path)
        plt.close()
        
        print(f"✅ Visualización estática guardada: {static_net_path}")
        return net_path, static_net_path
    
    return None, None

def process_city(city_name):
    print(f"\n{'='*50}")
    print(f"🏙️ PROCESANDO CIUDAD: {city_name}")
    print(f"{'='*50}")

    post_folder = f"{city_name}_CORPUS_LIMPIO"
    desc_folder = f"{city_name}_DESC_TRADUCIDO_LIMPIO"

    path_post = os.path.join(POST_FINALES_PATH, post_folder, f"analisis_{post_folder}.csv")
    path_desc = os.path.join(DESCRIPCIONES_FINALES_PATH, desc_folder, f"analisis_{desc_folder}.csv")

    print(f"Buscando posts en: {path_post}")
    print(f"Buscando descripciones en: {path_desc}")

    if not (os.path.exists(path_post) and os.path.exists(path_desc)):
        print(f"⚠️ Archivo(s) faltante(s) para {city_name}, saltando análisis.")
        return

    df_post = pd.read_csv(path_post)
    df_desc = pd.read_csv(path_desc)

    df_post['fuente'] = 'post'
    df_desc['fuente'] = 'descripcion'

    # Aplicar análisis individuales (sentimiento, frames)
    tqdm.pandas(desc="Analizando sentimiento en posts")
    df_post['sentimiento'] = df_post['texto_procesado'].progress_apply(analyze_sentiment)
    tqdm.pandas(desc="Analizando sentimiento en descripciones")
    df_desc['sentimiento'] = df_desc['texto_procesado'].progress_apply(analyze_sentiment)

    tqdm.pandas(desc="Detectando marcos en posts")
    df_post['frame'] = df_post['texto_procesado'].progress_apply(detect_frames)
    tqdm.pandas(desc="Detectando marcos en descripciones")
    df_desc['frame'] = df_desc['texto_procesado'].progress_apply(detect_frames)

    # Modelado de subtemas para cada conjunto
    df_post = bertopic_subtopics(df_post)
    df_desc = bertopic_subtopics(df_desc)

    # Generar reportes avanzados
    report_post_path, heatmap_post_path = generate_advanced_report(df_post, city_name, "posts")
    report_desc_path, heatmap_desc_path = generate_advanced_report(df_desc, city_name, "descripciones")

    # Construcción de red comparativa
    net_path, static_net_path = build_comparative_network(df_post, df_desc, city_name)

    # Guardar datasets analizados separados y combinados
    output_dir = os.path.join(ANALISIS_PATH, city_name)
    os.makedirs(output_dir, exist_ok=True)
    df_post.to_csv(os.path.join(output_dir, f"post_analizado_{city_name}.csv"), index=False)
    df_desc.to_csv(os.path.join(output_dir, f"desc_analizado_{city_name}.csv"), index=False)
    df_combined = pd.concat([df_post, df_desc], ignore_index=True)
    df_combined.to_csv(os.path.join(output_dir, f"combinado_analizado_{city_name}.csv"), index=False)

    print(f"💾 Datos analizados guardados en: {output_dir}")

    return report_post_path, report_desc_path, net_path, static_net_path

if __name__ == "__main__":
    cities = []
    if os.path.exists(POST_FINALES_PATH):
        # Extrae nombres únicos de ciudades desde los nombres de carpetas en POST_FINALES_PATH
        cities = list(set(
            [d.split('_')[0] for d in os.listdir(POST_FINALES_PATH) if os.path.isdir(os.path.join(POST_FINALES_PATH, d))]
        ))

    print("\nCiudades detectadas para análisis:")
    for idx, city in enumerate(sorted(cities), 1):
        print(f"{idx}. {city}")

    for city in sorted(cities):
        process_city(city)

    print("\n🚀 Análisis comparativo completado!")