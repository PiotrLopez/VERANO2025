import re
import unicodedata
from typing import Optional, List, Dict, Tuple
# from googletrans import Translator
from deep_translator import GoogleTranslator
import argparse
import pandas as pd
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import spacy
from collections import Counter
import json
import ssl
import time
from random import uniform
# Configurar SSL para NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Descargar recursos de NLTK si no están disponibles
recursos_nltk = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
for recurso in recursos_nltk:
    try:
        if recurso == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif recurso == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        elif recurso == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif recurso == 'wordnet':
            nltk.data.find('corpora/wordnet')
        elif recurso == 'omw-1.4':
            nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print(f"Descargando recurso NLTK: {recurso}")
        try:
            nltk.download(recurso)
        except Exception as e:
            print(f"Error descargando {recurso}: {e}")

class ProcesadorUnificado:
    def __init__(self):
        # Inicializar traductor con manejo de errores
        # try:
        try:
            # Prueba inicial de traducción
            test = GoogleTranslator(source='en', target='es').translate("ruins chichena maya temples beautiful ancient")
            print("test:",test)
            if test != "Hola, mi nombre es Julian, soy de la Ciudad de México y me gustan los videojuegos":
                print("⚠️ El traductor no está funcionando correctamente")
                self.traductor = None
            else:
                self.traductor = GoogleTranslator  # Clase, no instancia aún
        except Exception as e:
            print(f"⚠️ Fallo en prueba de traducción: {e}")
            self.traductor = None
        # Inicializar herramientas de NLP
        self.stemmer_es = SnowballStemmer('spanish')
        self.stemmer_en = SnowballStemmer('english')
        
        # Cargar modelo de spaCy para español
        try:
            self.nlp_es = spacy.load("es_core_news_sm")
        except OSError:
            print("⚠️  Modelo de spaCy no encontrado. Instalando...")
            print("   Ejecuta: python -m spacy download es_core_news_sm")
            self.nlp_es = None
        
        # Stopwords personalizadas
        try:
            self.stopwords_es = set(stopwords.words('spanish'))
            self.stopwords_en = set(stopwords.words('english'))
        except Exception as e:
            print(f"⚠️ Error cargando stopwords: {e}")
            self.stopwords_es = set()
            self.stopwords_en = set()
        
        # Agregar stopwords específicas de redes sociales
        stopwords_redes_sociales = {
            'rt', 'via', 'cc', 'ff', 'dm', 'mt', 'hh', 'pm', 'am',
            'follow', 'sigueme', 'like', 'share', 'comparte', 'siguenos',
            'instagram', 'facebook', 'twitter', 'tiktok', 'youtube',
            'post', 'publicacion', 'foto', 'video', 'story', 'stories',
            'live', 'directo', 'streaming', 'link', 'enlace', 'bio',
            'swipe', 'desliza', 'tap', 'toca', 'click', 'haz', 'clic',
            'image', 'picture', 'photo', 'showing', 'shows', 'show',
            'background', 'foreground', 'wearing', 'wear', 'color',
            'white', 'black', 'blue', 'red', 'green', 'yellow'
        }
        
        self.stopwords_es.update(stopwords_redes_sociales)
        self.stopwords_en.update(stopwords_redes_sociales)
        
        # Diccionario ampliado para corregir UTF-8
        self.correcciones_utf8 = {
            # Vocales con acentos
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã±': 'ñ', 'Ã': 'Á', 'Ã‰': 'É', 'Ã"': 'Í', 'Ã"': 'Ó', 'Ãš': 'Ú',
            'Å¡': 'š', 'Å¾': 'ž', 'Ã§': 'ç', 'Ã¼': 'ü', 'Ã¶': 'ö', 'Ã¤': 'ä',
            'Ã­a': 'ía', 'Ã©s': 'és', 'Ã¡s': 'ás', 'Ã³n': 'ón',
            'Ã¡r': 'ár', 'Ã©r': 'ér', 'Ã­r': 'ír', 'Ã³r': 'ór', 'Ãºr': 'úr',
            'Ã¡n': 'án', 'Ã©n': 'én', 'Ã­n': 'ín', 'Ã³n': 'ón', 'Ãºn': 'ún',
            'Ã¡l': 'ál', 'Ã©l': 'él', 'Ã­l': 'íl', 'Ã³l': 'ól', 'Ãºl': 'úl',
            # Comillas y caracteres especiales
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '–', 'â€"': '—',
            'Â': ' ', 'Â°': '°', 'Â¿': '¿', 'Â¡': '¡'
        }
    
    def corregir_codificacion_utf8(self, texto: str) -> str:
        """Corrige problemas de codificación UTF-8"""
        if not isinstance(texto, str):
            return str(texto)
        
        # Aplicar correcciones múltiples pasadas
        for _ in range(3):
            texto_anterior = texto
            for incorrecto, correcto in self.correcciones_utf8.items():
                texto = texto.replace(incorrecto, correcto)
            if texto == texto_anterior:
                break
        
        # Normalizar caracteres Unicode
        texto = unicodedata.normalize('NFC', texto)
        
        # Limpiar caracteres de control
        texto = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', texto)
        
        return texto
    
    def extraer_elementos_redes_sociales(self, texto: str) -> Dict[str, List[str]]:
        """Extrae hashtags, menciones y otros elementos"""
        elementos = {
            'hashtags': [],
            'menciones': [],
            'urls': [],
            'emojis': []
        }
        
        # Extraer hashtags
        hashtags = re.findall(r'#([a-zA-ZÀ-ÿ0-9_]+)', texto)
        elementos['hashtags'] = hashtags
        
        # Extraer menciones
        menciones = re.findall(r'@([a-zA-Z0-9_\.]+)', texto)
        elementos['menciones'] = menciones
        
        # Extraer URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', texto)
        elementos['urls'] = urls
        
        # Extraer emojis
        patron_emoji = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 u"\U0001F900-\U0001F9FF"  # Supplemental Symbols
                                 u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                 "]+", flags=re.UNICODE)
        emojis = patron_emoji.findall(texto)
        elementos['emojis'] = emojis
        
        return elementos
    
    def segmentar_hashtags(self, hashtag: str) -> List[str]:
        """Segmenta hashtags compuestos"""
        hashtag_limpio = hashtag.replace('#', '')
        
        # Separar por mayúsculas (CamelCase)
        palabras = re.findall(r'[A-Z][a-z]*|[a-z]+', hashtag_limpio)
        
        # Si no hay CamelCase, intentar separar por números
        if len(palabras) <= 1:
            palabras = re.findall(r'[a-zA-ZÀ-ÿ]+|\d+', hashtag_limpio)
        
        return [palabra.lower() for palabra in palabras if len(palabra) > 1]
    
    def limpiar_texto(self, texto: str, mantener_emojis: bool = True) -> str:
        """Limpia texto removiendo elementos no deseados"""
        if not isinstance(texto, str):
            return ""
        
        # Corregir encoding
        texto = self.corregir_codificacion_utf8(texto)
        
        # Remover URLs
        texto = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', texto)
        texto = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', texto)
        
        # Remover menciones
        texto = re.sub(r'@[a-zA-Z0-9_\.]+', '', texto)
        
        # Remover hashtags (se procesarán por separado)
        texto = re.sub(r'#[a-zA-ZÀ-ÿ0-9_]+', '', texto)
        
        # Remover emojis si se especifica
        if not mantener_emojis:
            patron_emoji = re.compile("["
                                     u"\U0001F600-\U0001F64F"
                                     u"\U0001F300-\U0001F5FF"
                                     u"\U0001F680-\U0001F6FF"
                                     u"\U0001F1E0-\U0001F1FF"
                                     u"\U00002702-\U000027B0"
                                     u"\U000024C2-\U0001F251"
                                     u"\U0001F900-\U0001F9FF"
                                     u"\U0001FA70-\U0001FAFF"
                                     "]+", flags=re.UNICODE)
            texto = patron_emoji.sub('', texto)
        
        # Limpiar caracteres especiales
        texto = re.sub(r'[^\w\s\.\,\;\:\!\?\¡\¿\-àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ\U0001F600-\U0001F6FF]', ' ', texto)
        
        # Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto
    import asyncio

    # async def traducir_texto(self, texto: str, idioma_destino: str = 'es') -> str:
    #     if not texto or not isinstance(texto, str):
    #         return ""
        
    #     if self.traductor is None:
    #         print("⚠️ Traductor no disponible, devolviendo texto original")
    #         return texto
        
    #     try:
    #         longitud_maxima = 4500
    #         if len(texto) > longitud_maxima:
    #             fragmentos = [texto[i:i+longitud_maxima] for i in range(0, len(texto), longitud_maxima)]
    #             fragmentos_traducidos = []
    #             for fragmento in fragmentos:
    #                 resultado = await self.traductor.translate(fragmento, src='auto', dest=idioma_destino)
    #                 if hasattr(resultado, 'text'):
    #                     fragmentos_traducidos.append(resultado.text)
    #                 else:
    #                     fragmentos_traducidos.append(fragmento)
    #             return ' '.join(fragmentos_traducidos)
    #         else:
    #             resultado = await self.traductor.translate(texto, src='auto', dest=idioma_destino)
    #             if hasattr(resultado, 'text'):
    #                 return resultado.text
    #             else:
    #                 return texto
    #     except Exception as e:
    #         print(f"Error en traducción: {e}")
    #         return texto
    def traducir_texto(self, texto: str, idioma_destino: str = 'es') -> str:
        """Traduce texto usando deep-translator con manejo de errores mejorado"""

        traductor = GoogleTranslator(source='en', target="es").translate(texto)

        resultado = traductor
        # print(traductor)
        return resultado



        
        
    # def lematizar_texto(self, texto: str, idioma: str = 'es') -> List[str]:
    #     """Aplica lematización usando spaCy o stemming como fallback"""
    #     if not texto:
    #         return []
        
    #     try:
    #         if idioma == 'es' and self.nlp_es:
    #             doc = self.nlp_es(texto)
    #             return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    #         else:
    #             # Fallback a stemming
    #             tokens = word_tokenize(texto.lower())
    #             if idioma == 'es':
    #                 return [self.stemmer_es.stem(token) for token in tokens if token.isalpha()]
    #             else:
    #                 return [self.stemmer_en.stem(token) for token in tokens if token.isalpha()]
    #     except Exception as e:
    #         print(f"Error en lematización: {e}")
    #         # Fallback básico: solo dividir por espacios y limpiar
    #         return [palabra.lower() for palabra in texto.split() if palabra.isalpha()]
    def lematizar_texto(self, texto: str, idioma: str = 'es') -> List[str]:
        """Aplica lematización usando spaCy con fallback básico (sin stemming)"""
        if not texto:
            return []
        
        try:
            if idioma == 'es' and self.nlp_es:
                doc = self.nlp_es(texto)
                return [token.lemma_.lower() for token in doc 
                        if not token.is_stop and not token.is_punct and token.is_alpha]
            else:
                # Fallback sin stemming: solo tokenización y limpieza básica
                tokens = word_tokenize(texto.lower())
                return [token for token in tokens if token.isalpha()]
                
        except Exception as e:
            print(f"Error en lematización: {e}")
            # Fallback básico: solo dividir por espacios y limpiar
            return [palabra.lower() for palabra in texto.split() if palabra.isalpha()]
    
    def eliminar_stopwords(self, tokens: List[str], idioma: str = 'es') -> List[str]:
        """Elimina stopwords de tokens"""
        if idioma == 'es':
            stopwords_conjunto = self.stopwords_es
        else:
            stopwords_conjunto = self.stopwords_en
        
        return [token for token in tokens if token.lower() not in stopwords_conjunto and len(token) > 2]
    
    # def procesar_texto_completo(self, texto: str, traducir: bool = False, 
    #                            procesar_hashtags: bool = True, languaje = "es") -> Dict:
    def procesar_texto_completo(self, texto: str, traducir: bool = False, 
                            procesar_hashtags: bool = True, languaje: str = "es") -> Dict:
        """Pipeline completo de procesamiento (método síncrono)"""
        
        # Paso 1: Extraer elementos de redes sociales
        elementos = self.extraer_elementos_redes_sociales(texto) if procesar_hashtags else {}
        
        # Paso 2: Limpiar texto
        texto_limpio = self.limpiar_texto(texto, mantener_emojis=True)
        
        # Paso 3: Traducir si es necesario (sin await)
        if traducir and texto_limpio:
            texto_limpio = self.traducir_texto(texto_limpio, languaje)
        
        # Paso 4: Aplicar lematización
        if texto_limpio:
            tokens_lematizados = self.lematizar_texto(texto_limpio, languaje)
            tokens_sin_stopwords = self.eliminar_stopwords(tokens_lematizados, languaje)
            texto_procesado = ' '.join(tokens_sin_stopwords)
        else:
            texto_procesado = ""
        
        # Paso 5: Procesar hashtags si existen
        hashtags_procesados = []
        if procesar_hashtags and elementos.get('hashtags'):
            for hashtag in elementos['hashtags']:
                palabras_hashtag = self.segmentar_hashtags(hashtag)
                # Aplicar lematización a palabras de hashtags
                for palabra in palabras_hashtag:
                    tokens_hashtag = self.lematizar_texto(palabra, languaje)
                    tokens_hashtag_limpios = self.eliminar_stopwords(tokens_hashtag, languaje)
                    hashtags_procesados.extend(tokens_hashtag_limpios)
        
        # Combinar texto procesado con hashtags procesados
        if hashtags_procesados:
            texto_final = f"{texto_procesado} {' '.join(hashtags_procesados)}".strip()
        else:
            texto_final = texto_procesado
        
        return {
            'texto_original': texto,
            'texto_procesado': texto_final,
            'longitud_original': len(texto),
            'longitud_procesada': len(texto_final),
            'hashtags': elementos.get('hashtags', []),
            'hashtags_procesados': hashtags_procesados,
            'menciones': elementos.get('menciones', []),
            'emojis': elementos.get('emojis', []),
            'urls': elementos.get('urls', [])
        }

    
    def analizar_tendencias(self, textos_procesados: List[str]) -> Dict:
        """Analiza tendencias en textos procesados"""
        todas_palabras = []
        for texto in textos_procesados:
            if texto:
                palabras = texto.split()
                todas_palabras.extend(palabras)
        
        contador_palabras = Counter(todas_palabras)
        
        return {
            'palabras_mas_frecuentes': contador_palabras.most_common(30),
            'vocabulario_unico': len(contador_palabras),
            'total_palabras': len(todas_palabras),
            'promedio_palabras_por_texto': len(todas_palabras) / len(textos_procesados) if textos_procesados else 0
        }
    
    # def procesar_datasets_unificados(self, archivo_clip: str, archivo_posts: str, 
    #                                archivo_salida: str, limite_muestras: int = 500) -> pd.DataFrame:
    def procesar_datasets_unificados(self, archivo_clip: str, archivo_posts: str, 
                               archivo_salida: str, limite_muestras: int = 500) -> pd.DataFrame:
        """Procesa ambos datasets de forma unificada (método síncrono)"""
        
        try:
            # Cargar datasets
            print(f"📂 Cargando datasets...")
            print(f"   • CLIP: {archivo_clip}")
            print(f"   • Instagram: {archivo_posts}")
            
            df_clip = pd.read_csv(archivo_clip, encoding='latin1')
            df_posts = pd.read_csv(archivo_posts, encoding='latin1')
            
            # Verificar columnas
            if 'prompt' not in df_clip.columns:
                print(f"❌ Error: Columna 'prompt' no encontrada en {archivo_clip}")
                print(f"   Columnas disponibles: {list(df_clip.columns)}")
                return None
            
            if 'description' not in df_posts.columns:
                print(f"❌ Error: Columna 'description' no encontrada en {archivo_posts}")
                print(f"   Columnas disponibles: {list(df_posts.columns)}")
                return None
            
            # Determinar número de muestras
            num_muestras_final = min(len(df_clip), len(df_posts), limite_muestras)
            
            print(f"\n📊 Información de procesamiento:")
            print(f"   • Muestras CLIP: {len(df_clip)}")
            print(f"   • Muestras Instagram: {len(df_posts)}")
            print(f"   • Límite configurado: {limite_muestras}")
            print(f"   • Muestras a procesar: {num_muestras_final}")
            
            # Tomar muestras
            df_clip_muestra = df_clip.head(num_muestras_final).copy()
            df_posts_muestra = df_posts.head(num_muestras_final).copy()
            
            print(f"\n🔄 Procesando textos con lematización y eliminación de stopwords...")
            
            # Procesar descripciones CLIP (sin await)
            print("   • Procesando descripciones CLIP (traducción + lematización)...")
            resultados_clip = []
            for i, descripcion in enumerate(df_clip_muestra['prompt']):
                if i % 100 == 0:
                    print(f"     Procesando CLIP {i+1}/{num_muestras_final}")
                
                if pd.isna(descripcion):
                    descripcion = ""
                
                try:
                    resultado = self.procesar_texto_completo(
                        str(descripcion), 
                        traducir=True, 
                        procesar_hashtags=False,
                        languaje="es"
                    )
                    resultados_clip.append(resultado)
                except Exception as e:
                    print(f"Error procesando CLIP {i+1}: {e}")
                    resultados_clip.append({
                        'texto_original': str(descripcion),
                        'texto_procesado': "",
                        'longitud_original': len(str(descripcion)),
                        'longitud_procesada': 0,
                        'hashtags': [],
                        'hashtags_procesados': [],
                        'menciones': [],
                        'emojis': [],
                        'urls': []
                    })
            
            # Procesar posts Instagram (sin await)
            print("   • Procesando posts Instagram (lematización + hashtags)...")
            resultados_posts = []
            for i, post in enumerate(df_posts_muestra['description']):
                if i % 100 == 0:
                    print(f"     Procesando Instagram {i+1}/{num_muestras_final}")
                
                if pd.isna(post):
                    post = ""
                
                try:
                    resultado = self.procesar_texto_completo(
                        str(post), 
                        traducir=True, 
                        procesar_hashtags=True,
                        languaje="es"
                    )
                    resultados_posts.append(resultado)
                except Exception as e:
                    print(f"Error procesando Instagram {i+1}: {e}")
                    resultados_posts.append({
                        'texto_original': str(post),
                        'texto_procesado': "",
                        'longitud_original': len(str(post)),
                        'longitud_procesada': 0,
                        'hashtags': [],
                        'hashtags_procesados': [],
                        'menciones': [],
                        'emojis': [],
                        'urls': []
                    })
            
            # Crear DataFrame final
            df_final = pd.DataFrame({
                'indice': range(num_muestras_final),
                'clip_original': [r['texto_original'] for r in resultados_clip],
                'clip_procesado': [r['texto_procesado'] for r in resultados_clip],
                'clip_longitud_original': [r['longitud_original'] for r in resultados_clip],
                'clip_longitud_procesada': [r['longitud_procesada'] for r in resultados_clip],
                'instagram_original': [r['texto_original'] for r in resultados_posts],
                'instagram_procesado': [r['texto_procesado'] for r in resultados_posts],
                'instagram_longitud_original': [r['longitud_original'] for r in resultados_posts],
                'instagram_longitud_procesada': [r['longitud_procesada'] for r in resultados_posts],
                'instagram_hashtags': [r['hashtags'] for r in resultados_posts],
                'instagram_hashtags_procesados': [r['hashtags_procesados'] for r in resultados_posts],
                'instagram_menciones': [r['menciones'] for r in resultados_posts],
                'instagram_emojis': [r['emojis'] for r in resultados_posts]
            })
            
            # Guardar resultado
            df_final.to_csv(archivo_salida, index=False)
            
            # Análisis de tendencias
            textos_clip = [r['texto_procesado'] for r in resultados_clip if r['texto_procesado']]
            textos_instagram = [r['texto_procesado'] for r in resultados_posts if r['texto_procesado']]
            
            tendencias_clip = self.analizar_tendencias(textos_clip)
            tendencias_instagram = self.analizar_tendencias(textos_instagram)
            
            print(f"\n✅ Procesamiento completado!")
            print(f"   📊 Estadísticas CLIP:")
            print(f"   • Vocabulario único: {tendencias_clip['vocabulario_unico']} palabras")
            print(f"   • Promedio palabras/texto: {tendencias_clip['promedio_palabras_por_texto']:.1f}")
            print(f"   • Top 5 palabras: {tendencias_clip['palabras_mas_frecuentes'][:5]}")
            
            print(f"\n   📊 Estadísticas Instagram:")
            print(f"   • Vocabulario único: {tendencias_instagram['vocabulario_unico']} palabras")
            print(f"   • Promedio palabras/texto: {tendencias_instagram['promedio_palabras_por_texto']:.1f}")
            print(f"   • Top 5 palabras: {tendencias_instagram['palabras_mas_frecuentes'][:5]}")
            
            print(f"\n   💾 Archivo guardado: {archivo_salida}")
            
            return df_final
                
        except Exception as e:
            print(f"❌ Error durante el procesamiento: {e}")
            import traceback
            traceback.print_exc()
            return None
    
# async def async_main():
#         parser = argparse.ArgumentParser(description='Procesador unificado CLIP e Instagram con lematización')
#         parser.add_argument('--clip', '-c', required=True, 
#                         help='Archivo CSV con descripciones CLIP (columna "prompt")')
#         parser.add_argument('--instagram', '-i', required=True, 
#                         help='Archivo CSV con posts Instagram (columna "description")')
#         parser.add_argument('--salida', '-s', required=True, 
#                         help='Archivo CSV de salida con ambos textos procesados')
#         parser.add_argument('--limite', '-l', type=int, default=500, 
#                         help='Número máximo de muestras a procesar (default: 500)')
        
#         args = parser.parse_args()
        
#         # Inicializar procesador
#         procesador = ProcesadorUnificado()
        
#         # Procesar datasets (con await)
#         resultado = await procesador.procesar_datasets_unificados(
#             archivo_clip=args.clip,
#             archivo_posts=args.instagram,
#             archivo_salida=args.salida,
#             limite_muestras=args.limite
#         )
        
#         if resultado is not None:
#             print(f"\n📋 Vista previa del resultado:")
#             columnas_muestra = ['clip_procesado', 'instagram_procesado', 'instagram_hashtags_procesados']
#             print(resultado[columnas_muestra].head(3))
#         else:
#             print("❌ El procesamiento falló. Revisa los errores anteriores.")

def main():
    parser = argparse.ArgumentParser(description='Procesador unificado CLIP e Instagram con lematización')
    parser.add_argument('--clip', '-c', required=True, 
                    help='Archivo CSV con descripciones CLIP (columna "prompt")')
    parser.add_argument('--instagram', '-i', required=True, 
                    help='Archivo CSV con posts Instagram (columna "description")')
    parser.add_argument('--salida', '-s', required=True, 
                    help='Archivo CSV de salida con ambos textos procesados')
    parser.add_argument('--limite', '-l', type=int, default=500, 
                    help='Número máximo de muestras a procesar (default: 500)')
    
    args = parser.parse_args()
    
    # Inicializar procesador
    procesador = ProcesadorUnificado()
    
    # Procesar datasets (sin await)
    resultado = procesador.procesar_datasets_unificados(
        archivo_clip=args.clip,
        archivo_posts=args.instagram,
        archivo_salida=args.salida,
        limite_muestras=args.limite
    )
    
    if resultado is not None:
        print(f"\n📋 Vista previa del resultado:")
        columnas_muestra = ['clip_procesado', 'instagram_procesado', 'instagram_hashtags_procesados']
        print(resultado[columnas_muestra].head(3))
    else:
        print("❌ El procesamiento falló. Revisa los errores anteriores.")
# Ejemplo de uso
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("🔧 Ejemplos de uso:")
        print("python procesador_unificado.py --clip descripciones_clip.csv --instagram posts_instagram.csv --salida dataset_procesado.csv")
        print("\nCon límite personalizado:")
        print("python procesador_unificado.py -c clip.csv -i instagram.csv -s resultado.csv -l 1000")
        print("\n📝 Requisitos:")
        print("- pip install googletrans==4.0.0rc1 pandas nltk spacy")
        print("- python -m spacy download es_core_news_sm")
        
        # Ejemplo de procesamiento individual
        print("\n🧪 Ejemplo de procesamiento individual:")
        try:
            procesador = ProcesadorUnificado()
            
            # Ejemplo CLIP
            texto_clip = "A person wearing a red shirt standing in front of a blue background"
            resultado_clip = procesador.procesar_texto_completo(texto_clip, traducir=True, procesar_hashtags=False)
            print(f"\n📷 CLIP Original: {resultado_clip['texto_original']}")
            print(f"📷 CLIP Procesado: {resultado_clip['texto_procesado']}")
            
            # Ejemplo Instagram
            texto_instagram = "¡Hermoso atardecer en #Acapulco! 🌅 @hotelencantoacapulco #ViajaConOrgullo #VeranoEnAcapulco"
            resultado_instagram = procesador.procesar_texto_completo(texto_instagram, traducir=False, procesar_hashtags=True)
            print(f"\n📱 Instagram Original: {resultado_instagram['texto_original']}")
            print(f"📱 Instagram Procesado: {resultado_instagram['texto_procesado']}")
            print(f"📱 Hashtags procesados: {resultado_instagram['hashtags_procesados']}")
            
        except Exception as e:
            print(f"Error en ejemplo: {e}")











