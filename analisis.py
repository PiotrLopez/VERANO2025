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

CORPUS_CLEAN_PATH = "Corpus_clean\\Rivera_Maya"
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

def bertopic_subtopics(df, text_column):
    """Aplica BERTopic para descubrir subtemas dentro de clusters"""
    print(f"🔍 Aplicando BERTopic para subtemas usando columna '{text_column}'...")
    if 'cluster' not in df.columns:
        print("⚠️ No se encontró columna 'cluster'. Saltando BERTopic.")
        return df
    
    df['subtopico'] = -1
    
    for cluster_id in tqdm(df['cluster'].unique(), desc="Modelando subtemas por cluster"):
        cluster_mask = df['cluster'] == cluster_id
        cluster_texts = df[cluster_mask][text_column].dropna().tolist()
        
        # Filtrar textos vacíos o muy cortos
        cluster_texts = [text for text in cluster_texts if len(str(text).strip()) > 10]
        
        if len(cluster_texts) > 5:
            try:
                topic_model = BERTopic(
                    embedding_model=sentence_model,
                    language="spanish",
                    nr_topics="auto",
                    verbose=False,
                    min_topic_size=2  # Reducir tamaño mínimo de tópico
                )
                
                topics, probabilities = topic_model.fit_transform(cluster_texts)
                
                # Verificar que topics tenga la longitud correcta
                if len(topics) == len(cluster_texts):
                    # Crear un mapeo para asignar subtópicos solo a los textos válidos
                    valid_indices = df[cluster_mask][text_column].dropna().index
                    valid_texts_indices = [i for i, text in enumerate(df.loc[valid_indices, text_column]) 
                                         if len(str(text).strip()) > 10]
                    
                    if len(valid_texts_indices) == len(topics):
                        for i, topic in enumerate(topics):
                            df.loc[valid_indices.iloc[valid_texts_indices[i]], 'subtopico'] = topic
                    else:
                        print(f"⚠️ Desajuste en índices para cluster {cluster_id}, asignando -1")
                        df.loc[cluster_mask, 'subtopico'] = -1
                else:
                    print(f"⚠️ Longitud de topics no coincide para cluster {cluster_id}")
                    df.loc[cluster_mask, 'subtopico'] = -1
                    
            except Exception as e:
                print(f"⚠️ Error en cluster {cluster_id}: {str(e)}")
                df.loc[cluster_mask, 'subtopico'] = -1
        else:
            print(f"📝 Cluster {cluster_id} tiene muy pocos textos ({len(cluster_texts)}), saltando subtópicos")
            df.loc[cluster_mask, 'subtopico'] = -1
    
    return df

def extract_cluster_keywords(df, cluster_col='cluster', text_col='texto_procesado', top_n=10):
    """Extrae palabras clave por cluster usando TF-IDF"""
    cluster_keywords = {}
    for cluster_id in df[cluster_col].unique():
        cluster_texts = df[df[cluster_col] == cluster_id][text_col].dropna().tolist()
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

def generate_advanced_report(df, city_name, source_type, text_col):
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
    cluster_keywords = extract_cluster_keywords(df, text_col=text_col)
    
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
        cluster_texts = df[df['cluster'] == cluster_id][text_col].dropna().tolist()
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
        keywords = extract_cluster_keywords(cluster_df, text_col='descripcion_procesada_es').get(cl, "Sin palabras clave")
        
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
        keywords = extract_cluster_keywords(cluster_df, text_col='prompt_procesado_es').get(cl, "Sin palabras clave")
        
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
            post_texts = df_post[df_post['cluster'] == post_cl]['descripcion_procesada_es'].dropna().tolist()
            desc_texts = df_desc[df_desc['cluster'] == desc_cl]['prompt_procesado_es'].dropna().tolist()
            
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

def apply_clustering(df, text_column, n_clusters=10):
    """Aplica clustering usando embeddings de sentence transformer"""
    print(f"🔍 Aplicando clustering a columna '{text_column}'...")
    texts = df[text_column].dropna().tolist()
    
    if len(texts) < 2:
        print("⚠️ No hay suficientes textos para clustering")
        df['cluster'] = 0
        return df
    
    # Generar embeddings
    embeddings = sentence_model.encode(texts)
    
    # Aplicar clustering (usando KMeans como ejemplo)
    from sklearn.cluster import KMeans
    n_clusters = min(n_clusters, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Asignar clusters al DataFrame
    df['cluster'] = -1  # Inicializar
    valid_indices = df[text_column].dropna().index
    df.loc[valid_indices, 'cluster'] = clusters
    
    return df

def process_city(city_name):
    print(f"\n{'='*50}")
    print(f"🏙️ PROCESANDO CIUDAD: {city_name}")
    print(f"{'='*50}")

    csv_path = os.path.join(CORPUS_CLEAN_PATH, f"{city_name}.csv")
    
    print(f"Buscando archivo CSV en: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"⚠️ Archivo CSV no encontrado para {city_name}, saltando análisis.")
        return

    # Cargar el CSV
    df = pd.read_csv(csv_path)
    
    # Verificar que las columnas necesarias existan
    if 'descripcion_procesada_es' not in df.columns or 'prompt_procesado_es' not in df.columns:
        print(f"⚠️ Columnas requeridas no encontradas en {city_name}.csv")
        print(f"Columnas disponibles: {list(df.columns)}")
        return

    # Separar posts y descripciones
    df_post = df[['descripcion_procesada_es']].copy()
    df_post = df_post.dropna(subset=['descripcion_procesada_es'])
    df_post['fuente'] = 'post'
    
    df_desc = df[['prompt_procesado_es']].copy()
    df_desc = df_desc.dropna(subset=['prompt_procesado_es'])
    df_desc['fuente'] = 'descripcion'

    if len(df_post) == 0 or len(df_desc) == 0:
        print(f"⚠️ No hay datos suficientes para análisis en {city_name}")
        return

    # Aplicar clustering
    df_post = apply_clustering(df_post, 'descripcion_procesada_es')
    df_desc = apply_clustering(df_desc, 'prompt_procesado_es')

    # Aplicar análisis individuales (sentimiento, frames)
    tqdm.pandas(desc="Analizando sentimiento en posts")
    df_post['sentimiento'] = df_post['descripcion_procesada_es'].progress_apply(analyze_sentiment)
    tqdm.pandas(desc="Analizando sentimiento en descripciones")
    df_desc['sentimiento'] = df_desc['prompt_procesado_es'].progress_apply(analyze_sentiment)

    tqdm.pandas(desc="Detectando marcos en posts")
    df_post['frame'] = df_post['descripcion_procesada_es'].progress_apply(detect_frames)
    tqdm.pandas(desc="Detectando marcos en descripciones")
    df_desc['frame'] = df_desc['prompt_procesado_es'].progress_apply(detect_frames)

    # Modelado de subtemas para cada conjunto
    df_post = bertopic_subtopics(df_post, 'descripcion_procesada_es')
    df_desc = bertopic_subtopics(df_desc, 'prompt_procesado_es')

    # Generar reportes avanzados
    report_post_path, heatmap_post_path = generate_advanced_report(df_post, city_name, "posts", 'descripcion_procesada_es')
    report_desc_path, heatmap_desc_path = generate_advanced_report(df_desc, city_name, "descripciones", 'prompt_procesado_es')

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
    if os.path.exists(CORPUS_CLEAN_PATH):
        # Extrae nombres de ciudades desde los archivos CSV en CORPUS_CLEAN_PATH
        cities = [f.replace('.csv', '') for f in os.listdir(CORPUS_CLEAN_PATH) 
                 if f.endswith('.csv') and os.path.isfile(os.path.join(CORPUS_CLEAN_PATH, f))]

    print("\nCiudades detectadas para análisis:")
    for idx, city in enumerate(sorted(cities), 1):
        print(f"{idx}. {city}")

    for city in sorted(cities):
        process_city(city)

    print("\n🚀 Análisis comparativo completado!")