import os
from preprocesamiento import ProcesadorUnificado  # Asumiendo que tu clase está en este módulo

# Configuración de rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPCIONES_DIR = os.path.join(BASE_DIR, "Descripciones")
IMAGENES_DIR = os.path.join(BASE_DIR, "Extraccion_imagenes")
CORPUS_DIR = os.path.join(BASE_DIR, "Corpus_clean")

# Lista de ciudades (puedes obtenerlas dinámicamente si lo prefieres)
CIUDADES = [
    "CDMX", 
    "MONTERREY",
    "CANCUN",
    "GUADALAJARA",
    "ACAPULCO",
    "MERIDA",
    "MAZATLAN",
    "PUEBLA",
    # Agrega todas tus ciudades aquí
]

def procesar_ciudad(ciudad: str, limite: int = 500):
    """Procesa los archivos para una ciudad específica"""
    try:
        # Construye los nombres de archivo (asumiendo el patrón ISLA<CIUDAD>)
        nombre_base = f"{ciudad.replace(' ', '').upper()}"
        
        # Construye las rutas completas
        archivo_clip = os.path.join(DESCRIPCIONES_DIR,  f"{nombre_base}_desc.csv")
        archivo_instagram = os.path.join(IMAGENES_DIR,  f"{nombre_base}.csv")
        archivo_salida = os.path.join(CORPUS_DIR, f"{nombre_base}_clean.csv")
        
        # Crea directorios de salida si no existen
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Procesando ciudad: {ciudad}")
        print(f"📂 CLIP: {archivo_clip}")
        print(f"📂 Instagram: {archivo_instagram}")
        print(f"💾 Salida: {archivo_salida}")
        
        # Procesa los archivos
        procesador = ProcesadorUnificado()
        resultado = procesador.procesar_datasets_unificados(
            archivo_clip=archivo_clip,
            archivo_posts=archivo_instagram,
            archivo_salida=archivo_salida,
            limite_muestras=limite
        )
        
        if resultado is not None:
            print(f"✅ {ciudad} procesada correctamente")
        else:
            print(f"❌ Error al procesar {ciudad}")
            
    except Exception as e:
        print(f"❌ Error crítico al procesar {ciudad}: {str(e)}")

def main():
    # Procesa todas las ciudades
    for ciudad in CIUDADES:
        procesar_ciudad(ciudad)
    
    print("\nProcesamiento completado para todas las ciudades")

if __name__ == "__main__":
    main()