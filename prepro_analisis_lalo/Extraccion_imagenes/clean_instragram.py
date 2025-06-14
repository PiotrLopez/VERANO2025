import os
import re
import pandas as pd
from unidecode import unidecode
import wordninja  # Para dividir palabras concatenadas

def limpiar_texto(texto):
    """Limpia el texto de una publicación de Instagram."""
    if not isinstance(texto, str):
        return ""
    
    # Elimina emojis
    texto = re.sub(r'[\U00010000-\U0010ffff]', '', texto, flags=re.UNICODE)
    
    # Elimina URLs
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    
    # Elimina menciones (@usuario)
    texto = re.sub(r'@\w+', '', texto)
    
    # Convierte hashtags a texto legible y separado
    def split_hashtag(match):
        tag = match.group(1)
        # Divide palabras concatenadas (ej: ciudaddelasmontañas → ciudad de las montanas)
        words = wordninja.split(tag.lower())
        return ' '.join(words)
    
    texto = re.sub(r'#(\w+)', split_hashtag, texto)
    
    # Elimina signos de puntuación (excepto espacios)
    texto = re.sub(r'[^\w\s]', '', texto)
    
    # Quita acentos de vocales usando unidecode
    texto = unidecode(texto)
    
    # Minúsculas y trim
    texto = texto.strip().lower()
    
    return texto

# Ruta donde están los archivos CSV
ruta_input = '/home/trigu3roslalo/Documentos/LIDIA - Universidad de Guanajuato/6° Semestre/XXX_Verano_de_la_Ciencia/verano/Extraccion_imagenes'

# Carpeta de salida para los archivos limpios
ruta_output = 'publicacion_corpus_limpio'
os.makedirs(ruta_output, exist_ok=True)

# Archivos a procesar con sus codificaciones
archivos = [
    {'path': os.path.join(ruta_input, 'CDMX.csv'), 'encoding': 'utf-8'},
    {'path': os.path.join(ruta_input, 'GUADALAJARA.csv'), 'encoding': 'utf-8'},
    {'path': os.path.join(ruta_input, 'MONTERREY.csv'), 'encoding': 'latin1'}
]

# Verifica si los archivos existen
archivos_existentes = [archivo for archivo in archivos if os.path.exists(archivo['path'])]
archivos_faltantes = [archivo for archivo in archivos if not os.path.exists(archivo['path'])]

if archivos_faltantes:
    print("⚠️ Advertencia: Los siguientes archivos no se encontraron:")
    for archivo in archivos_faltantes:
        print(f"  ❌ {archivo['path']}")
    print("Procesando solo los archivos existentes...\n")

for archivo in archivos_existentes:
    try:
        print(f"📄 Procesando archivo: {archivo['path']}")
        
        # Lee el CSV con la codificación correcta
        df = pd.read_csv(archivo['path'], on_bad_lines='skip', low_memory=False, encoding=archivo['encoding'])
        
        # Verifica que exista la columna 'description'
        if 'description' not in df.columns:
            print(f"⚠️ La columna 'description' no se encontró en {archivo['path']}. Saltando...")
            continue

        # Limpia el texto
        df['texto_original'] = df['description']
        df['texto_limpio'] = df['description'].apply(limpiar_texto)

        # Guarda solo las columnas relevantes
        df_salida = df[['texto_original', 'texto_limpio']]

        # Nombre del archivo de salida
        nombre_base = os.path.basename(archivo['path'])
        nombre_salida = os.path.join(ruta_output, f"{nombre_base}_corpus_limpio.csv")

        # Guarda el nuevo CSV en UTF-8
        df_salida.to_csv(nombre_salida, index=False, encoding='utf-8')
        print(f"✅ Archivo guardado: {nombre_salida}\n")

    except Exception as e:
        print(f"❌ Error procesando {archivo['path']}: {e}\n")