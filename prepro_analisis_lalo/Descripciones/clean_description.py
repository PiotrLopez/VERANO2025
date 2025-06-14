import os
import re
import pandas as pd
from unidecode import unidecode
from googletrans import Translator
from tqdm import tqdm  # Para barra de progreso

# Configura tqdm para pandas
tqdm.pandas()

# Función para traducir texto de inglés a español
def traducir_texto(texto):
    try:
        translator = Translator()
        traduccion = translator.translate(texto, src='en', dest='es')
        return traduccion.text
    except Exception as e:
        print(f"❌ Error al traducir: {e}")
        return texto  # Devuelve el texto original si falla

# Función para limpiar el texto (sin acentos, minúsculas, sin signos)
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    # Elimina signos de puntuación y convierte a minúsculas
    texto = re.sub(r'[^\w\s]', '', texto).lower()
    # Elimina acentos usando unidecode
    texto = unidecode(texto)
    return texto.strip()

# Ruta donde están los archivos CSV
ruta_input = '/home/trigu3roslalo/Documentos/LIDIA - Universidad de Guanajuato/6° Semestre/XXX_Verano_de_la_Ciencia/verano/Descripciones'

# Carpeta de salida para los archivos limpios
ruta_output = 'clean_descripciones'
os.makedirs(ruta_output, exist_ok=True)

# Archivos a procesar
archivos = [
    os.path.join(ruta_input, 'CDMX_desc.csv'),
    os.path.join(ruta_input, 'GUADALAJARA_desc.csv'),
    os.path.join(ruta_input, 'MONTERREY_desc.csv')  # Nota: archivo con typo en el nombre
]

# Verifica si los archivos existen
archivos_existentes = []
for archivo in archivos:
    if os.path.exists(archivo):
        archivos_existentes.append(archivo)
    else:
        print(f"⚠️ Archivo no encontrado: {archivo}")

if not archivos_existentes:
    print("❌ No se encontró ningún archivo CSV. Terminando ejecución.")
    exit()

for archivo in archivos_existentes:
    try:
        print(f"📄 Procesando archivo: {archivo}")
        
        # Lee el CSV con UTF-8
        df = pd.read_csv(archivo, on_bad_lines='skip', low_memory=False, encoding='utf-8')
        
        # Asegura que la columna sea 'prompt' (ajusta si es necesario)
        if 'prompt' not in df.columns:
            print(f"⚠️ La columna 'prompt' no se encontró en {archivo}. Saltando...")
            continue

        # Muestra el número de filas a procesar
        total_filas = len(df)
        print(f"🔄 Traduciendo y limpiando {total_filas} filas...")

        # Traduce el texto con barra de progreso
        print("🌐 Traduciendo descripciones...")
        df['prompt_traducido'] = df['prompt'].progress_apply(traducir_texto)
        
        # Limpia el texto traducido con barra de progreso
        print("🧹 Limpiando descripciones...")
        df['prompt_limpio'] = df['prompt_traducido'].progress_apply(limpiar_texto)

        # Guarda solo las columnas relevantes
        df_salida = df[['prompt', 'prompt_traducido', 'prompt_limpio']]

        # Nombre del archivo de salida
        nombre_base = os.path.basename(archivo)
        nombre_salida = os.path.join(ruta_output, f"{nombre_base}_traducido_limpio.csv")

        # Guarda el nuevo CSV en UTF-8
        df_salida.to_csv(nombre_salida, index=False, encoding='utf-8')
        print(f"✅ Archivo guardado: {nombre_salida}\n")

    except Exception as e:
        print(f"❌ Error procesando {archivo}: {e}\n")