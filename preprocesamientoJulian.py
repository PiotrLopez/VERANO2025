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
import wordninja
from deep_translator import GoogleTranslator
import os
from pathlib import Path
import glob
from datetime import datetime
import logging

# Configuración de logging con encoding UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('procesamiento_ciudades.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup NLTK
try:
    nltk.download('stopwords', quiet=True)
    toktok = ToktokTokenizer()
    stop_words = set(stopwords.words('english'))
    logger.info("OK NLTK configurado correctamente")
except Exception as e:
    logger.error(f"ERROR configurando NLTK: {e}")
    raise

# --- FUNCIONES ORIGINALES (con corrección) ---
def remove_stop_words(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def split_hashtag(tag):
    if tag.startswith('@'):
        return ""
    elif tag.startswith('#'):
        tag_content = tag[1:]
        if re.search(r'[A-Z]', tag_content):
            return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag_content)
        else:
            segmented = wordninja.split(tag_content)
            return " ".join(segmented)
    else:
        return tag

def remove_punctuation_mark(text, replace=" "):
    text = text.replace("\n", " ")
    def process_match(match):
        full_match = match.group(0)
        return split_hashtag(full_match)
    text = re.sub(r'[@#]\w+', process_match, text)
    return re.sub(r'[%s]' % re.escape(string.punctuation + '¡¿´©✕""''†•−˚'), replace, text)

def tokens(text):
    return toktok.tokenize(text)

def avoidVoid(text):
    return isinstance(text, str) and text.strip() != ""

def get_bigrams(words):
    return " ".join([f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)])

def remove_consecutive_duplicates(words):
    if not words:
        return []
    result = [words[0]]
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    return result

def remove_repeated_ngrams(text, max_ngram_size=5):
    for n in range(max_ngram_size, 1, -1):
        pattern = re.compile(rf'(\b(?:\w+)\b(?:\s+\b(?:\w+)\b){{{n-1}}})(?:\s+\1)+', flags=re.IGNORECASE)
        while True:
            new_text = pattern.sub(r'\1', text)
            if new_text == text:
                break
            text = new_text
    return text

def procesar_lista_textos(lista_textos, bigramas=0, max_doc_freq=0.05):
    """
    Procesa una lista de textos manteniendo el mismo número de elementos.
    CORREGIDO: Ahora preserva el orden y cantidad original de textos.
    """
    if not lista_textos:
        return []
    
    # Procesar cada texto individualmente
    textos_procesados = []
    doc_freq = defaultdict(int)
    total_docs = 0
    
    # Primera pasada: procesar textos y contar frecuencias
    for text in lista_textos:
        if avoidVoid(text):
            # Procesar texto
            text = clean(text, no_emoji=True)
            text = text.lower()
            text = remove_stop_words(text)
            text = remove_punctuation_mark(text)
            words = tokens(text)
            words = remove_consecutive_duplicates(words)

            if bigramas == 0:
                processed_text = " ".join(words)
            elif bigramas == 1:
                processed_text = get_bigrams(words)
            elif bigramas == 2:
                processed_text = " ".join(words) + " " + get_bigrams(words)
            
            textos_procesados.append(processed_text)
            
            # Contar frecuencias
            unique_words = set(processed_text.split())
            for word in unique_words:
                doc_freq[word] += 1
            total_docs += 1
        else:
            # Mantener texto vacío como vacío
            textos_procesados.append("")
    
    # Segunda pasada: filtrar palabras muy comunes
    common_words = {word for word, freq in doc_freq.items() if freq / total_docs > max_doc_freq}
    
    final_texts = []
    for text in textos_procesados:
        if text.strip():
            filtered_text = " ".join([word for word in text.split() if word not in common_words])
            final_texts.append(remove_repeated_ngrams(filtered_text))
        else:
            final_texts.append("")
    
    return final_texts

def traducir_textos(lista_textos, idioma_origen='es', idioma_destino='en', ciudad_nombre=""):
    """
    Traduce una lista de textos usando deep-translator
    """
    if not lista_textos:
        return []
    
    translator = GoogleTranslator(source=idioma_origen, target=idioma_destino)
    textos_traducidos = []
    
    logger.info(f"TRADUCCION [{ciudad_nombre}] Iniciando traduccion de {len(lista_textos)} textos ({idioma_origen} -> {idioma_destino})...")
    
    for i, texto in enumerate(lista_textos):
        try:
            if texto and isinstance(texto, str) and texto.strip():
                # Manejar textos largos
                if len(texto) > 4500:
                    # Dividir en partes más pequeñas
                    palabras = texto.split()
                    partes = []
                    parte_actual = []
                    longitud_actual = 0
                    
                    for palabra in palabras:
                        if longitud_actual + len(palabra) + 1 > 4500:
                            if parte_actual:
                                partes.append(" ".join(parte_actual))
                                parte_actual = [palabra]
                                longitud_actual = len(palabra)
                        else:
                            parte_actual.append(palabra)
                            longitud_actual += len(palabra) + 1
                    
                    if parte_actual:
                        partes.append(" ".join(parte_actual))
                    
                    # Traducir cada parte
                    traducciones_partes = []
                    for parte in partes:
                        traduccion_parte = translator.translate(parte)
                        traducciones_partes.append(traduccion_parte)
                    
                    traduccion = " ".join(traducciones_partes)
                else:
                    traduccion = translator.translate(texto)
                
                textos_traducidos.append(traduccion)
                
                # Mostrar progreso menos frecuente para reducir ruido
                if (i + 1) % 10 == 0:
                    logger.info(f"   [{ciudad_nombre}] OK Traducidos: {i+1}/{len(lista_textos)}")
                
            else:
                textos_traducidos.append("")  # Mantener vacíos como vacíos
                
        except Exception as e:
            logger.warning(f"   [{ciudad_nombre}] ADVERTENCIA Error traduciendo texto {i+1}: {e}")
            # En caso de error, mantener el texto original
            textos_traducidos.append(str(texto) if texto else "")
    
    logger.info(f"COMPLETADO [{ciudad_nombre}] Traduccion completada: {len(textos_traducidos)} textos procesados")
    return textos_traducidos

# --- FUNCIONES DE AUTOMATIZACIÓN ---
def encontrar_ciudades(directorio_descripciones, directorio_imagenes, region="Rivera_Maya"):
    """
    Encuentra todas las ciudades disponibles en ambos directorios
    """
    # Buscar archivos de descripciones
    patron_desc = os.path.join(directorio_descripciones, region, "*_desc.csv")
    archivos_desc = glob.glob(patron_desc)
    
    # Buscar archivos de imágenes
    patron_img = os.path.join(directorio_imagenes, region, "*.csv")
    archivos_img = glob.glob(patron_img)
    
    # Extraer nombres de ciudades
    ciudades_desc = set()
    for archivo in archivos_desc:
        nombre = os.path.basename(archivo).replace('_desc.csv', '')
        ciudades_desc.add(nombre)
    
    ciudades_img = set()
    for archivo in archivos_img:
        nombre = os.path.basename(archivo).replace('.csv', '')
        ciudades_img.add(nombre)
    
    # Ciudades que tienen ambos archivos
    ciudades_completas = ciudades_desc.intersection(ciudades_img)
    
    logger.info(f"CIUDADES Ciudades encontradas:")
    logger.info(f"   • Con descripciones: {len(ciudades_desc)} - {sorted(ciudades_desc)}")
    logger.info(f"   • Con imagenes: {len(ciudades_img)} - {sorted(ciudades_img)}")
    logger.info(f"   • Completas (ambos archivos): {len(ciudades_completas)} - {sorted(ciudades_completas)}")
    
    return sorted(ciudades_completas)

def procesar_ciudad(ciudad, config):
    """
    Procesa una ciudad específica
    CORREGIDO: Ahora verifica que todas las listas tengan la misma longitud
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESANDO CIUDAD: {ciudad}")
    logger.info(f"{'='*80}")
    
    # Construir rutas
    desc_path = os.path.join(config['dir_descripciones'], config['region'], f"{ciudad}_desc.csv")
    img_path = os.path.join(config['dir_imagenes'], config['region'], f"{ciudad}.csv")
    output_path = os.path.join(config['dir_salida'], f"{config['region']}{ciudad}_clean.csv")
    
    # Verificar que existan los archivos
    if not os.path.exists(desc_path):
        logger.error(f"ERROR [{ciudad}] No se encontro archivo de descripciones: {desc_path}")
        return False
    
    if not os.path.exists(img_path):
        logger.error(f"ERROR [{ciudad}] No se encontro archivo de imagenes: {img_path}")
        return False
    
    try:
        # Cargar archivos
        logger.info(f"ARCHIVOS [{ciudad}] Cargando archivos...")
        df_desc = pd.read_csv(desc_path, encoding='latin1')
        df_img = pd.read_csv(img_path, encoding='latin1')
        
        # Aplicar límite de muestras
        dim1, dim2 = df_desc.shape[0], df_img.shape[0]
        max_muestras = min(config['max_muestras'], dim1, dim2)
        
        df_desc = df_desc.head(max_muestras)
        df_img = df_img.head(max_muestras)
        
        logger.info(f"DATOS [{ciudad}] Procesando {max_muestras} registros (de {dim1} desc, {dim2} img)")
        
        # Extraer textos originales
        textos_desc_originales = df_desc['prompt'].fillna("").astype(str).tolist()
        textos_img_originales = df_img['description'].fillna("").astype(str).tolist()
        
        # Verificar que ambas listas tengan la misma longitud
        if len(textos_desc_originales) != len(textos_img_originales):
            min_length = min(len(textos_desc_originales), len(textos_img_originales))
            logger.warning(f"ADVERTENCIA [{ciudad}] Longitudes diferentes: desc={len(textos_desc_originales)}, img={len(textos_img_originales)}. Truncando a {min_length}")
            textos_desc_originales = textos_desc_originales[:min_length]
            textos_img_originales = textos_img_originales[:min_length]
        
        # PASO 1: Traducir textos
        logger.info(f"PASO1 [{ciudad}] PASO 1: Traduciendo textos...")
        textos_desc_traducidos = traducir_textos(
            textos_desc_originales, 
            config['idioma_origen'], 
            config['idioma_destino'],
            ciudad
        )
        textos_img_traducidos = traducir_textos(
            textos_img_originales, 
            config['idioma_origen'], 
            config['idioma_destino'],
            ciudad
        )
        
        # Verificar longitudes después de traducir
        logger.info(f"VERIFICACION [{ciudad}] Longitudes después de traducir:")
        logger.info(f"   • Originales desc: {len(textos_desc_originales)}")
        logger.info(f"   • Originales img: {len(textos_img_originales)}")
        logger.info(f"   • Traducidos desc: {len(textos_desc_traducidos)}")
        logger.info(f"   • Traducidos img: {len(textos_img_traducidos)}")
        
        # PASO 2: Procesar textos traducidos
        logger.info(f"PASO2 [{ciudad}] PASO 2: Procesando textos traducidos...")
        procesados_desc = procesar_lista_textos(
            textos_desc_traducidos, 
            bigramas=config['bigramas'], 
            max_doc_freq=config['max_doc_freq']
        )
        procesados_img = procesar_lista_textos(
            textos_img_traducidos, 
            bigramas=config['bigramas'], 
            max_doc_freq=config['max_doc_freq']
        )
        
        # Verificar longitudes después de procesar
        logger.info(f"VERIFICACION [{ciudad}] Longitudes después de procesar:")
        logger.info(f"   • Procesados desc: {len(procesados_desc)}")
        logger.info(f"   • Procesados img: {len(procesados_img)}")
        
        # Verificar que todas las listas tengan la misma longitud
        longitudes = [
            len(textos_desc_originales),
            len(textos_img_originales),
            len(textos_desc_traducidos),
            len(textos_img_traducidos),
            len(procesados_desc),
            len(procesados_img)
        ]
        
        if len(set(longitudes)) > 1:
            logger.error(f"ERROR [{ciudad}] Longitudes inconsistentes: {longitudes}")
            return False
        
        # PASO 3: Crear dataset final
        logger.info(f"PASO3 [{ciudad}] PASO 3: Creando dataset final...")
        df_combinado = pd.DataFrame({
            'prompt_original_en': textos_desc_originales,
            'descripcion_original_en': textos_img_originales,
            'prompt_traducido_es': textos_desc_traducidos,
            'descripcion_traducida_es': textos_img_traducidos,
            'prompt_procesado_es': procesados_desc,
            'descripcion_procesada_es': procesados_img
        })
        
        # Crear directorio de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar archivo
        df_combinado.to_csv(output_path, sep=',', index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        
        # Estadísticas
        textos_desc_vacios = sum(1 for x in textos_desc_traducidos if not str(x).strip())
        textos_img_vacios = sum(1 for x in textos_img_traducidos if not str(x).strip())
        
        logger.info(f"EXITO [{ciudad}] COMPLETADO:")
        logger.info(f"   • Registros procesados: {len(df_combinado)}")
        logger.info(f"   • Textos vacios desc: {textos_desc_vacios}")
        logger.info(f"   • Textos vacios img: {textos_img_vacios}")  
        logger.info(f"   • Archivo guardado: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR [{ciudad}] Error procesando ciudad: {e}")
        import traceback
        logger.error(f"TRACEBACK [{ciudad}] {traceback.format_exc()}")
        return False

def procesar_todas_ciudades(config):
    """
    Procesa todas las ciudades encontradas
    """
    logger.info(f"\nINICIO PROCESAMIENTO AUTOMATIZADO")
    logger.info(f"FECHA Y HORA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Encontrar ciudades
    ciudades = encontrar_ciudades(
        config['dir_descripciones'], 
        config['dir_imagenes'], 
        config['region']
    )
    
    if not ciudades:
        logger.error("ERROR No se encontraron ciudades para procesar")
        return
    
    # Filtrar ciudades si se especifica
    if config.get('ciudades_especificas'):
        ciudades_filtradas = [c for c in ciudades if c in config['ciudades_especificas']]
        logger.info(f"FILTRO Procesando solo ciudades especificas: {ciudades_filtradas}")
        ciudades = ciudades_filtradas
    
    # Procesar cada ciudad
    total_ciudades = len(ciudades)
    ciudades_exitosas = 0
    ciudades_fallidas = []
    
    for i, ciudad in enumerate(ciudades, 1):
        logger.info(f"\nPROCESANDO Procesando ciudad {i}/{total_ciudades}: {ciudad}")
        
        exito = procesar_ciudad(ciudad, config)
        
        if exito:
            ciudades_exitosas += 1
            logger.info(f"EXITO {ciudad} procesada exitosamente")
        else:
            ciudades_fallidas.append(ciudad)
            logger.error(f"ERROR {ciudad} fallo en el procesamiento")
    
    # Resumen final
    logger.info(f"\n{'='*80}")
    logger.info(f"RESUMEN FINAL DEL PROCESAMIENTO")
    logger.info(f"{'='*80}")
    logger.info(f"TOTAL Total de ciudades: {total_ciudades}")
    logger.info(f"EXITOSAS Exitosas: {ciudades_exitosas}")
    logger.info(f"FALLIDAS Fallidas: {len(ciudades_fallidas)}")
    
    if ciudades_fallidas:
        logger.warning(f"ADVERTENCIA Ciudades que fallaron: {ciudades_fallidas}")
    
    logger.info(f"FINALIZADO Procesamiento completado!")

# --- CONFIGURACIÓN PRINCIPAL ---
def main():
    """
    Función principal para ejecutar el procesamiento
    """
    # Configuración personalizable
    config = {
        # Directorios
        'dir_descripciones': r'C:\Users\pumgu\VERANO2025\Descripciones',
        'dir_imagenes': r'C:\Users\pumgu\VERANO2025\Extraccion_imagenes',
        'dir_salida': r'C:\Users\pumgu\VERANO2025\Corpus_clean',
        
        # Parámetros de procesamiento
        'region': 'Rivera_Maya',
        'max_muestras': 500,  # Límite de registros por ciudad
        'idioma_origen': 'en',
        'idioma_destino': 'es',
        'bigramas': 0,
        'max_doc_freq': 0.1,
        
        # Ciudades específicas (opcional - dejar vacío para procesar todas)
        'ciudades_especificas': ["CHICHENITZA", "UXMAL"],  # Ejemplo: ['CHETUMAL', 'CANCUN']
    }
    
    # Ejecutar procesamiento
    procesar_todas_ciudades(config)

# --- FUNCIONES AUXILIARES ---
def listar_ciudades_disponibles():
    """
    Lista todas las ciudades disponibles sin procesarlas
    """
    config = {
        'dir_descripciones': r'C:\Users\pumgu\VERANO2025\Descripciones',
        'dir_imagenes': r'C:\Users\pumgu\VERANO2025\Extraccion_imagenes',
        'region': 'Rivera_Maya',
    }
    
    ciudades = encontrar_ciudades(
        config['dir_descripciones'], 
        config['dir_imagenes'], 
        config['region']
    )
    
    print(f"\nCIUDADES Ciudades disponibles para procesar:")
    for i, ciudad in enumerate(ciudades, 1):
        print(f"   {i:2d}. {ciudad}")
    
    return ciudades

def procesar_ciudades_especificas(lista_ciudades):
    """
    Procesa solo las ciudades especificadas en la lista
    """
    config = {
        'dir_descripciones': r'C:\Users\pumgu\VERANO2025\Descripciones',
        'dir_imagenes': r'C:\Users\pumgu\VERANO2025\Extraccion_imagenes',
        'dir_salida': r'C:\Users\pumgu\VERANO2025\Corpus_clean',
        'region': 'Rivera_Maya',
        'max_muestras': 500,
        'idioma_origen': 'en',
        'idioma_destino': 'es',
        'bigramas': 0,
        'max_doc_freq': 0.1,
        'ciudades_especificas': lista_ciudades,
    }
    
    procesar_todas_ciudades(config)

if __name__ == "__main__":
    # Opciones de ejecución:
    
    # 1. Listar ciudades disponibles
    # listar_ciudades_disponibles()
    
    # 2. Procesar todas las ciudades
    main()
    
    # 3. Procesar ciudades específicas
    # procesar_ciudades_especificas(['CHETUMAL', 'CANCUN'])





# import numpy as np
# import pandas as pd
# import re
# import csv
# import nltk
# import string
# from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.corpus import stopwords
# from cleantext import clean
# from collections import defaultdict
# import wordninja
# from deep_translator import GoogleTranslator  # pip install deep-translator

# # Setup
# nltk.download('stopwords')
# toktok = ToktokTokenizer()
# stop_words = set(stopwords.words('english'))

# # --- Funciones originales ---
# def remove_stop_words(text):
#     return " ".join([word for word in text.split() if word.lower() not in stop_words])

# def split_hashtag(tag):
#     if tag.startswith('@'):
#         return ""
#     elif tag.startswith('#'):
#         tag_content = tag[1:]
#         if re.search(r'[A-Z]', tag_content):
#             return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag_content)
#         else:
#             segmented = wordninja.split(tag_content)
#             return " ".join(segmented)
#     else:
#         return tag

# def remove_punctuation_mark(text, replace=" "):
#     text = text.replace("\n", " ")
#     def process_match(match):
#         full_match = match.group(0)
#         return split_hashtag(full_match)
#     text = re.sub(r'[@#]\w+', process_match, text)
#     return re.sub(r'[%s]' % re.escape(string.punctuation + '¡¿´©✕""''†•−˚'), replace, text)

# def tokens(text):
#     return toktok.tokenize(text)

# def avoidVoid(text):
#     return isinstance(text, str) and text.strip() != ""

# def get_bigrams(words):
#     return " ".join([f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)])

# def remove_consecutive_duplicates(words):
#     if not words:
#         return []
#     result = [words[0]]
#     for word in words[1:]:
#         if word != result[-1]:
#             result.append(word)
#     return result

# def remove_repeated_ngrams(text, max_ngram_size=5):
#     for n in range(max_ngram_size, 1, -1):
#         pattern = re.compile(rf'(\b(?:\w+)\b(?:\s+\b(?:\w+)\b){{{n-1}}})(?:\s+\1)+', flags=re.IGNORECASE)
#         while True:
#             new_text = pattern.sub(r'\1', text)
#             if new_text == text:
#                 break
#             text = new_text
#     return text

# def procesar_lista_textos(lista_textos, bigramas=0, max_doc_freq=0.05):
#     new_df = []
#     list_df = []
#     doc_freq = defaultdict(int)
#     total_docs = 0

#     for text in lista_textos:
#         if avoidVoid(text):
#             text = clean(text, no_emoji=True)
#             text = text.lower()
#             text = remove_stop_words(text)
#             text = remove_punctuation_mark(text)
#             words = tokens(text)
#             words = remove_consecutive_duplicates(words)

#             if bigramas == 0:
#                 processed_text = " ".join(words)
#             elif bigramas == 1:
#                 processed_text = get_bigrams(words)
#             elif bigramas == 2:
#                 processed_text = " ".join(words) + " " + get_bigrams(words)

#             list_df.append(processed_text)
#             unique_words = set(processed_text.split())
#             for word in unique_words:
#                 doc_freq[word] += 1
#             total_docs += 1

#     common_words = {word for word, freq in doc_freq.items() if freq / total_docs > max_doc_freq}

#     final_texts = []
#     for text in list_df:
#         filtered_text = " ".join([word for word in text.split() if word not in common_words])
#         final_texts.append(remove_repeated_ngrams(filtered_text))

#     return final_texts

# # --- FUNCIÓN DE TRADUCCIÓN MEJORADA ---
# def traducir_textos(lista_textos, idioma_origen='es', idioma_destino='en'):
#     """
#     Traduce una lista de textos usando deep-translator
#     """
#     translator = GoogleTranslator(source=idioma_origen, target=idioma_destino)
#     textos_traducidos = []
    
#     print(f"🌍 Iniciando traducción de {len(lista_textos)} textos ({idioma_origen} → {idioma_destino})...")
    
#     for i, texto in enumerate(lista_textos):
#         try:
#             if texto and isinstance(texto, str) and texto.strip():
#                 # Manejar textos largos
#                 if len(texto) > 4500:
#                     # Dividir en partes más pequeñas
#                     palabras = texto.split()
#                     partes = []
#                     parte_actual = []
#                     longitud_actual = 0
                    
#                     for palabra in palabras:
#                         if longitud_actual + len(palabra) + 1 > 4500:
#                             if parte_actual:
#                                 partes.append(" ".join(parte_actual))
#                                 parte_actual = [palabra]
#                                 longitud_actual = len(palabra)
#                         else:
#                             parte_actual.append(palabra)
#                             longitud_actual += len(palabra) + 1
                    
#                     if parte_actual:
#                         partes.append(" ".join(parte_actual))
                    
#                     # Traducir cada parte
#                     traducciones_partes = []
#                     for parte in partes:
#                         traduccion_parte = translator.translate(parte)
#                         traducciones_partes.append(traduccion_parte)
                    
#                     traduccion = " ".join(traducciones_partes)
#                 else:
#                     traduccion = translator.translate(texto)
                
#                 textos_traducidos.append(traduccion)
                
#                 # Mostrar progreso
#                 if (i + 1) % 5 == 0 or i == 0:
#                     print(f"   ✓ Traducidos: {i+1}/{len(lista_textos)}")
#                     print(f"     Original: '{str(texto)[:50]}...'")
#                     print(f"     Traducido: '{str(traduccion)[:50]}...'")
                
#             else:
#                 textos_traducidos.append("")  # Mantener vacíos como vacíos
                
#         except Exception as e:
#             print(f"   ⚠️ Error traduciendo texto {i+1}: {e}")
#             # En caso de error, mantener el texto original
#             textos_traducidos.append(str(texto) if texto else "")
    
#     print(f"✅ Traducción completada: {len(textos_traducidos)} textos procesados")
#     return textos_traducidos

# # --- Rutas ---
# desc_path = r'C:\Users\pumgu\VERANO2025\Descripciones\Rivera_Maya\CHETUMAL_desc.csv'
# img_path = r'C:\Users\pumgu\VERANO2025\Extraccion_imagenes\Rivera_Maya\CHETUMAL.csv'
# output_path = r'C:\Users\pumgu\VERANO2025\Corpus_clean\Rivera_MayaCHETUMAL_clean.csv'

# # --- Carga archivos ---
# print("📂 Cargando archivos...")
# df_desc = pd.read_csv(desc_path)
# df_img = pd.read_csv(img_path)

# dim1 = df_desc.shape[0]
# dim2 = df_img.shape[0]
# # --- Limitar a muestras para prueba ---
# max_muestras = min(500, dim1, dim2)
# print(max_muestras)
# df_desc = df_desc.head(max_muestras)
# df_img = df_img.head(max_muestras)

# print(df_desc.shape)
# print(df_img.shape)
# print(f"📊 Procesando {max_muestras} registros...")

# # --- Extraer textos originales ---
# print("📝 Extrayendo textos originales...")
# textos_desc_originales = df_desc['prompt'].fillna("").astype(str).tolist()
# textos_img_originales = df_img['description'].fillna("").astype(str).tolist()

# print(f"   • Descripciones: {len(textos_desc_originales)} textos")
# print(f"   • Imágenes: {len(textos_img_originales)} textos")

# # --- PASO 1: TRADUCIR TEXTOS ORIGINALES ---
# print("\n" + "="*60)
# print("PASO 1: TRADUCIENDO TEXTOS ORIGINALES (ES → EN)")
# print("="*60)

# textos_desc_traducidos = traducir_textos(textos_desc_originales, 'en', 'es')
# textos_img_traducidos = traducir_textos(textos_img_originales, 'en', 'es')

# # --- PASO 2: PROCESAR TEXTOS TRADUCIDOS ---
# print("\n" + "="*60)
# print("PASO 2: PROCESANDO TEXTOS TRADUCIDOS")
# print("="*60)

# print("🧹 Procesando descripciones traducidas...")
# procesados_desc = procesar_lista_textos(textos_desc_traducidos, bigramas=0, max_doc_freq=0.1)

# print("🧹 Procesando imágenes traducidas...")
# procesados_img = procesar_lista_textos(textos_img_traducidos, bigramas=0, max_doc_freq=0.1)

# print("✅ Procesamiento completado")

# # --- PASO 3: CREAR DATASET FINAL ---
# print("\n" + "="*60)
# print("PASO 3: CREANDO DATASET FINAL")
# print("="*60)

# df_combinado = pd.DataFrame({
#     'prompt_original_en': textos_desc_originales,
#     'descripcion_original_en': textos_img_originales,
#     'prompt_traducido_es': textos_desc_traducidos,
#     'descripcion_traducida_es': textos_img_traducidos,
#     'prompt_procesado_es': procesados_desc,
#     'descripcion_procesada_es': procesados_img
# })

# # Guardar archivo
# df_combinado.to_csv(output_path, sep=',', index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
# print(f"💾 Archivo guardado en: {output_path}")

# # --- ESTADÍSTICAS FINALES ---
# print(f"\n📈 ESTADÍSTICAS FINALES:")
# print(f"   • Registros procesados: {len(df_combinado)}")
# print(f"   • Columnas creadas: {len(df_combinado.columns)}")
# print(f"   • Textos vacíos en desc traducidas: {sum(1 for x in textos_desc_traducidos if not str(x).strip())}")
# print(f"   • Textos vacíos en img traducidas: {sum(1 for x in textos_img_traducidos if not str(x).strip())}")

# # --- MOSTRAR EJEMPLOS ---
# print(f"\n🔍 EJEMPLOS DE RESULTADOS:")
# print("="*60)
# for i in range(min(3, len(df_combinado))):
#     print(f"\n📝 EJEMPLO {i+1}:")
#     print(f"   ES original: '{textos_desc_originales[i][:60]}...'")
#     print(f"   EN traducido: '{textos_desc_traducidos[i][:60]}...'")
#     print(f"   EN procesado: '{procesados_desc[i][:60]}...'")
#     print("-" * 60)

# print(f"\n✅ ¡PROCESO COMPLETADO EXITOSAMENTE!")
# print(f"📁 Revisa el archivo: {output_path}")

# import numpy as np
# import pandas as pd
# import re
# import csv
# import nltk
# import string
# from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.corpus import stopwords
# from cleantext import clean
# from collections import defaultdict
# import wordninja
# from deep_translator import GoogleTranslator  # pip install deep-translator

# # Setup
# nltk.download('stopwords')
# toktok = ToktokTokenizer()
# stop_words = set(stopwords.words('english'))

# # --- Funciones originales ---
# def remove_stop_words(text):
#     return " ".join([word for word in text.split() if word.lower() not in stop_words])

# def split_hashtag(tag):
#     if tag.startswith('@'):
#         return ""
#     elif tag.startswith('#'):
#         tag_content = tag[1:]
#         if re.search(r'[A-Z]', tag_content):
#             return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag_content)
#         else:
#             segmented = wordninja.split(tag_content)
#             return " ".join(segmented)
#     else:
#         return tag

# def remove_punctuation_mark(text, replace=" "):
#     text = text.replace("\n", " ")
#     def process_match(match):
#         full_match = match.group(0)
#         return split_hashtag(full_match)
#     text = re.sub(r'[@#]\w+', process_match, text)
#     return re.sub(r'[%s]' % re.escape(string.punctuation + '¡¿´©✕""''†•−˚'), replace, text)

# def tokens(text):
#     return toktok.tokenize(text)

# def avoidVoid(text):
#     return isinstance(text, str) and text.strip() != ""

# def get_bigrams(words):
#     return " ".join([f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)])

# def remove_consecutive_duplicates(words):
#     if not words:
#         return []
#     result = [words[0]]
#     for word in words[1:]:
#         if word != result[-1]:
#             result.append(word)
#     return result

# def remove_repeated_ngrams(text, max_ngram_size=5):
#     for n in range(max_ngram_size, 1, -1):
#         pattern = re.compile(rf'(\b(?:\w+)\b(?:\s+\b(?:\w+)\b){{{n-1}}})(?:\s+\1)+', flags=re.IGNORECASE)
#         while True:
#             new_text = pattern.sub(r'\1', text)
#             if new_text == text:
#                 break
#             text = new_text
#     return text

# def procesar_lista_textos(lista_textos, bigramas=0, max_doc_freq=0.05):
#     new_df = []
#     list_df = []
#     doc_freq = defaultdict(int)
#     total_docs = 0

#     for text in lista_textos:
#         if avoidVoid(text):
#             text = clean(text, no_emoji=True)
#             text = text.lower()
#             text = remove_stop_words(text)
#             text = remove_punctuation_mark(text)
#             words = tokens(text)
#             words = remove_consecutive_duplicates(words)

#             if bigramas == 0:
#                 processed_text = " ".join(words)
#             elif bigramas == 1:
#                 processed_text = get_bigrams(words)
#             elif bigramas == 2:
#                 processed_text = " ".join(words) + " " + get_bigrams(words)

#             list_df.append(processed_text)
#             unique_words = set(processed_text.split())
#             for word in unique_words:
#                 doc_freq[word] += 1
#             total_docs += 1

#     common_words = {word for word, freq in doc_freq.items() if freq / total_docs > max_doc_freq}

#     final_texts = []
#     for text in list_df:
#         filtered_text = " ".join([word for word in text.split() if word not in common_words])
#         final_texts.append(remove_repeated_ngrams(filtered_text))

#     return final_texts

# # --- NUEVA FUNCIÓN DE TRADUCCIÓN ---
# def traducir_textos(lista_textos, idioma_origen='en', idioma_destino='es'):
#     """
#     Traduce una lista de textos usando deep-translator
    
#     Args:
#         lista_textos: Lista de textos a traducir
#         idioma_origen: Código del idioma origen ('es', 'en', etc.)
#         idioma_destino: Código del idioma destino ('en', 'es', etc.)
    
#     Returns:
#         Lista de textos traducidos
#     """
#     translator = GoogleTranslator(source=idioma_origen, target=idioma_destino)
#     textos_traducidos = []
    
#     print(f"🌍 Iniciando traducción de {len(lista_textos)} textos ({idioma_origen} → {idioma_destino})...")
    
#     for i, texto in enumerate(lista_textos):
#         try:
#             if texto and texto.strip():  # Solo traducir si no está vacío
#                 # Manejar textos largos (límite de deep-translator: 5000 chars)
#                 if len(texto) > 4500:
#                     # Dividir en partes más pequeñas
#                     palabras = texto.split()
#                     partes = []
#                     parte_actual = []
#                     longitud_actual = 0
                    
#                     for palabra in palabras:
#                         if longitud_actual + len(palabra) + 1 > 4500:
#                             if parte_actual:
#                                 partes.append(" ".join(parte_actual))
#                                 parte_actual = [palabra]
#                                 longitud_actual = len(palabra)
#                         else:
#                             parte_actual.append(palabra)
#                             longitud_actual += len(palabra) + 1
                    
#                     if parte_actual:
#                         partes.append(" ".join(parte_actual))
                    
#                     # Traducir cada parte
#                     traducciones_partes = []
#                     for parte in partes:
#                         print(parte)
#                         traducciones_partes.append(translator.translate(parte))
                    
#                     traduccion = " ".join(traducciones_partes)
#                 else:
#                     traduccion = translator.translate(texto)
                
#                 textos_traducidos.append(traduccion)
                
#                 # Mostrar progreso cada 10 textos
#                 if (i + 1) % 10 == 0 or i == 0:
#                     print(f"   ✓ Traducidos: {i+1}/{len(lista_textos)}")
#                     print(f"     Ejemplo: '{texto[:50]}...' → '{traduccion[:50]}...'")
                
#             else:
#                 textos_traducidos.append("")  # Mantener vacíos como vacíos
                
#         except Exception as e:
#             print(f"   ⚠️ Error traduciendo texto {i+1}: {e}")
#             textos_traducidos.append(texto)  # Mantener original si falla
    
#     print(f"✅ Traducción completada: {len(textos_traducidos)} textos procesados")
#     return textos_traducidos

# # --- Rutas ---
# desc_path = r'C:\Users\pumgu\VERANO2025\Descripciones\CANCUN_desc.csv'
# img_path = r'C:\Users\pumgu\VERANO2025\Extraccion_Imagenes\CANCUN.csv'
# output_path = r'C:\Users\pumgu\VERANO2025\Corpus_clean\CANCUN_clean_traducido.csv'

# # --- Carga archivos ---
# print("📂 Cargando archivos...")
# df_desc = pd.read_csv(desc_path)
# df_img = pd.read_csv(img_path)

# # --- Limitar a 500 registros (o menos) ---
# max_muestras = min(10, len(df_desc))
# df_desc = df_desc.head(max_muestras)
# df_img = df_img.head(max_muestras)

# print(f"📊 Procesando {max_muestras} registros...")

# # --- Preprocesamiento ---
# print("🧹 Iniciando preprocesamiento...")
# textos_desc = df_desc['prompt'].tolist()
# textos_img = df_img['description'].tolist()

# # Traducir descripciones procesadas
# procesados_desc_traducidos = traducir_textos(textos_desc, 'es', 'en')

# # Traducir imágenes procesadas  
# procesados_img_traducidos = traducir_textos(textos_img, 'es', 'en')

# procesados_desc = procesar_lista_textos(procesados_desc_traducidos, bigramas=0, max_doc_freq=0.1)
# procesados_img = procesar_lista_textos(procesados_img_traducidos, bigramas=0, max_doc_freq=0.1)

# print("✅ Preprocesamiento completado")

# # --- TRADUCCIÓN (NUEVO) ---
# print("\n" + "="*50)
# print("INICIANDO TRADUCCIÓN")
# print("="*50)



# # --- Combinar y guardar ---
# print("\n💾 Creando archivo final...")
# df_combinado = pd.DataFrame({
#     'prompt_original': textos_desc,
#     'descripcion_original': textos_img,
#     'prompt_traducido_procesado': procesados_desc,
#     'descripcion_traducida:_procesada': procesados_img
# })

# df_combinado.to_csv(output_path, sep=',', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
# print(f"✔ Archivo combinado con traducciones guardado en: {output_path}")

# # --- Estadísticas finales ---
# print(f"\n📈 ESTADÍSTICAS FINALES:")
# print(f"   • Registros procesados: {len(df_combinado)}")
# print(f"   • Columnas creadas: {len(df_combinado.columns)}")
# print(f"   • Textos vacíos en traducciones desc: {sum(1 for x in procesados_desc_traducidos if not x.strip())}")
# print(f"   • Textos vacíos en traducciones img: {sum(1 for x in procesados_img_traducidos if not x.strip())}")

# # Mostrar algunos ejemplos
# print(f"\n🔍 EJEMPLOS DE TRADUCCIÓN:")
# for i in range(min(3, len(procesados_desc))):
#     if procesados_desc[i].strip() and procesados_desc_traducidos[i].strip():
#         print(f"\nEjemplo {i+1}:")
#         print(f"   ES: {procesados_desc[i][:80]}...")
#         print(f"   EN: {procesados_desc_traducidos[i][:80]}...")

# import numpy as np
# import pandas as pd
# import re
# import csv
# import nltk
# import string
# from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.corpus import stopwords
# from cleantext import clean
# from collections import defaultdict
# import wordninja
# from deep_translator import GoogleTranslator

# # Setup
# nltk.download('stopwords')
# toktok = ToktokTokenizer()
# stop_words = set(stopwords.words('english'))

# # --- Funciones ---
# def remove_stop_words(text):
#     return " ".join([word for word in text.split() if word.lower() not in stop_words])

# def split_hashtag(tag):
#     """Función corregida para procesar hashtags y menciones"""
#     if tag.startswith('@'):
#         return ""  # Elimina menciones como @usuario
#     elif tag.startswith('#'):
#         tag_content = tag[1:]  # Elimina solo el símbolo #
#         if re.search(r'[A-Z]', tag_content):
#             # Separar tipo CamelCase: AmoMiVida → Amo Mi Vida
#             # print(tag_content)    
#             return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag_content)
        
#     else:
#         return tag  # Devuelve el texto tal cual si no empieza con # o @

# def remove_punctuation_mark(text, replace=" "):
#     """Función corregida para procesar hashtags correctamente"""
#     text = text.replace("\n", " ")
    
#     # Procesar hashtags y menciones ANTES de eliminar puntuación
#     def process_match(match):
#         full_match = match.group(0)  # Incluye el # o @
#         return split_hashtag(full_match)
    
#     # Aplicar el procesamiento a hashtags y menciones
#     text = re.sub(r'[@#]\w+', process_match, text)
    
#     # Ahora eliminar puntuación restante
#     return re.sub(r'[%s]' % re.escape(string.punctuation + '¡¿´©✕""''†•−˚'), replace, text)

# def tokens(text):
#     return toktok.tokenize(text)

# def avoidVoid(text):
#     return isinstance(text, str) and text.strip() != ""

# def get_bigrams(words):
#     return " ".join([f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)])

# def remove_consecutive_duplicates(words):
#     if not words:
#         return []
#     result = [words[0]]
#     for word in words[1:]:
#         if word != result[-1]:
#             result.append(word)
#     return result

# def remove_repeated_ngrams(text, max_ngram_size=5):
#     for n in range(max_ngram_size, 1, -1):
#         pattern = re.compile(rf'(\b(?:\w+)\b(?:\s+\b(?:\w+)\b){{{n-1}}})(?:\s+\1)+', flags=re.IGNORECASE)
#         while True:
#             new_text = pattern.sub(r'\1', text)
#             if new_text == text:
#                 break
#             text = new_text
#     return text

# # --- Función principal ---
# def procesar_lista_textos(lista_textos, bigramas=0, max_doc_freq=0.05):
#     new_df = []
#     list_df = []
#     doc_freq = defaultdict(int)
#     total_docs = 0

#     for text in lista_textos:
#         if avoidVoid(text):
#             text = clean(text, no_emoji=True)
#             text = text.lower()
#             text = remove_punctuation_mark(text)
#             text = remove_stop_words(text)
#             words = tokens(text)
#             words = remove_consecutive_duplicates(words)

#             if bigramas == 0:
#                 processed_text = " ".join(words)
#             elif bigramas == 1:
#                 processed_text = get_bigrams(words)
#             elif bigramas == 2:
#                 processed_text = " ".join(words) + " " + get_bigrams(words)

#             list_df.append(processed_text)
#             unique_words = set(processed_text.split())
#             for word in unique_words:
#                 doc_freq[word] += 1
#             total_docs += 1

#     common_words = {word for word, freq in doc_freq.items() if freq / total_docs > max_doc_freq}

#     final_texts = []
#     for text in list_df:
#         filtered_text = " ".join([word for word in text.split() if word not in common_words])
#         final_texts.append(remove_repeated_ngrams(filtered_text))

#     return final_texts

# def traducir_texto(texto: str, idioma_destino: str = 'es') -> str:
#         """Traduce texto usando deep-translator con manejo de errores mejorado"""

#         traductor = GoogleTranslator(source='en', target="es").translate(texto)

#         resultado = traductor
#         # print(traductor)
#         return resultado

# # --- Rutas ---
# desc_path = r'C:\Users\pumgu\VERANO2025\Descripciones\CANCUN_desc.csv'
# img_path = r'C:\Users\pumgu\VERANO2025\Extraccion_Imagenes\CANCUN.csv'
# output_path = r'C:\Users\pumgu\VERANO2025\Corpus_clean\CANCUN_clean.csv'

# # --- Carga archivos ---
# df_desc = pd.read_csv(desc_path)
# df_img = pd.read_csv(img_path)

# # --- Limitar a 500 registros (o menos) ---
# max_muestras = min(500, len(df_desc))
# df_desc = df_desc.head(max_muestras)
# df_img = df_img.head(max_muestras)

# # --- Preprocesamiento ---
# textos_desc = df_desc['prompt'].tolist()
# textos_img = df_img['description'].tolist()
# procesados_desc = procesar_lista_textos(textos_desc, bigramas=0, max_doc_freq=0.1)
# procesados_img = procesar_lista_textos(textos_img, bigramas=0, max_doc_freq=0.1)
# procesados_desc_traduccidos = traducir_texto(procesados_desc)
# # --- Combinar y guardar ---
# df_combinado = pd.DataFrame({
#     'prompt_preprocesado': procesados_desc,
#     'prompt_preprocesado_traducido': procesados_desc_traduccidos,
#     'descripcion_preprocesada': procesados_img
# })

# df_combinado.to_csv(output_path, sep=',', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
# print(f"✔ Archivo combinado guardado en: {output_path}")


# # import re
# # import wordninja

# # def split_hashtag(tag):
# #     tag = tag.strip('#@')

# #     # Si contiene al menos una letra mayúscula, separa por camel case
# #     if re.search(r'[A-Z]', tag):
# #         return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tag)

# #     # Si no hay mayúsculas pero venía de un hashtag, intenta separarlo igual como si fueran minúsculas
# #     if tag.islower() or tag.isalpha():  # Si todo minúscula o sin símbolos raros
# #         return " ".join(wordninja.split(tag))
    
# #     return tag  # Por defecto, retorna el tag sin procesar
# # def replace_hashtags_and_mentions(text):
# #     return re.sub(r'[@#]\w+', lambda m: split_hashtag(m.group()), text)

# # texto = "Amo viajar con @hotellasestrella y usar #AmoMiVida #viajafeliz"
# # print(replace_hashtags_and_mentions(texto))

