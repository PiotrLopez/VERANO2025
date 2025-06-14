import os

def find_csv_files(base_path):
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files

if __name__ == "__main__":
    post_finales_path = 'post_finales'
    descripciones_finales_path = 'descripciones_finales'

    print("Buscando archivos CSV en 'post_finales'...")
    post_csv_files = find_csv_files(post_finales_path)
    for path in post_csv_files:
        print(path)

    print("\nBuscando archivos CSV en 'descripciones_finales'...")
    desc_csv_files = find_csv_files(descripciones_finales_path)
    for path in desc_csv_files:
        print(path)
