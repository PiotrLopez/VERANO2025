import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class CorpusAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Corpus - Ciudades Turísticas México")
        self.root.geometry("800x800")
        
        # Variables
        self.data = None
        self.current_file = None
        self.current_column = None
        
        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_gui()
        
    def setup_gui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar peso de las columnas y filas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Panel de control
        control_frame = ttk.LabelFrame(main_frame, text="Control de Análisis", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Cargar archivo
        ttk.Label(control_frame, text="Archivo:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.file_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.file_var, width=50).grid(row=0, column=1, padx=(5, 0), pady=2)
        ttk.Button(control_frame, text="Cargar CSV", command=self.load_file).grid(row=0, column=2, padx=(5, 0), pady=2)
        
        # Seleccionar columna
        ttk.Label(control_frame, text="Columna:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(control_frame, textvariable=self.column_var, width=30)
        self.column_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Parámetros de n-gramas
        ttk.Label(control_frame, text="N-gramas (n):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.n_var = tk.StringVar(value="2")
        ttk.Entry(control_frame, textvariable=self.n_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Número máximo de palabras
        ttk.Label(control_frame, text="Max palabras:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.max_words_var = tk.StringVar(value="100")
        ttk.Entry(control_frame, textvariable=self.max_words_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Botones de análisis
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Nube de Palabras", command=self.generate_wordcloud).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Análisis N-gramas", command=self.analyze_ngrams).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Estadísticas", command=self.show_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Limpiar", command=self.clear_plots).pack(side=tk.LEFT, padx=5)
        
        # Panel de visualización
        viz_frame = ttk.LabelFrame(main_frame, text="Visualizaciones", padding="5")
        viz_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel de información
        info_frame = ttk.LabelFrame(main_frame, text="Información del Corpus", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.current_file = file_path
                self.file_var.set(os.path.basename(file_path))
                
                # Actualizar combo de columnas
                columns = [col for col in self.data.columns if 'procesado' in col.lower()]
                if not columns:
                    columns = list(self.data.columns)
                
                self.column_combo['values'] = columns
                if 'clip_procesado' in columns:
                    self.column_var.set('clip_procesado')
                elif 'instagram_procesado' in columns:
                    self.column_var.set('instagram_procesado')
                elif columns:
                    self.column_var.set(columns[0])
                
                self.show_file_info()
                messagebox.showinfo("Éxito", f"Archivo cargado: {len(self.data)} registros")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")
    
    def show_file_info(self):
        if self.data is not None:
            city_name = os.path.splitext(os.path.basename(self.current_file))[0]
            
            info = f"📍 CIUDAD: {city_name.upper()}\n"
            info += f"📊 REGISTROS TOTALES: {len(self.data)}\n"
            info += f"📋 COLUMNAS DISPONIBLES: {', '.join(self.data.columns)}\n\n"
            
            # Información por columna procesada
            for col in self.data.columns:
                if 'procesado' in col.lower():
                    non_null = self.data[col].notna().sum()
                    info += f"🔸 {col}: {non_null} registros válidos\n"
                    
                    # Muestra de datos
                    sample_data = self.data[col].dropna().head(3)
                    for i, text in enumerate(sample_data):
                        if isinstance(text, str) and len(text) > 0:
                            preview = text[:100] + "..." if len(text) > 100 else text
                            info += f"   Ejemplo {i+1}: {preview}\n"
                    info += "\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
    
    def preprocess_text(self, texts):
        """Preprocesa una serie de textos"""
        processed = []
        for text in texts:
            # if pd.isna(text) or not isinstance(text, str):
            #     continue
                
            # # Limpiar texto
            # text = text.lower()
            # text = re.sub(r'[^\w\s]', ' ', text)
            # text = re.sub(r'\s+', ' ', text)
            # text = text.strip()
            
            # if len(text) > 0:
            processed.append(text)
        
        return processed
    
    def generate_wordcloud(self):
        if not self.validate_inputs():
            return
            
        try:
            column = self.column_var.get()
            max_words = int(self.max_words_var.get())
            
            # Preprocesar textos
            texts = self.preprocess_text(self.data[column])
            
            if not texts:
                messagebox.showwarning("Advertencia", "No hay texto válido para procesar")
                return
            
            # Combinar todos los textos
            combined_text = ' '.join(texts)
            
            # Generar nube de palabras
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=max_words,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate(combined_text)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            
            city_name = os.path.splitext(os.path.basename(self.current_file))[0]
            ax.set_title(f'Nube de Palabras - {city_name} ({column})', fontsize=14, fontweight='bold')
            
            # Agregar a notebook
            self.add_plot_to_notebook(fig, f"WordCloud - {column}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar nube de palabras: {str(e)}")
    
    def analyze_ngrams(self):
        if not self.validate_inputs():
            return
            
        try:
            column = self.column_var.get()
            n = int(self.n_var.get())
            max_words = int(self.max_words_var.get())
            
            # Preprocesar textos
            texts = self.preprocess_text(self.data[column])
            
            if not texts:
                messagebox.showwarning("Advertencia", "No hay texto válido para procesar")
                return
            
            # Generar n-gramas
            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                max_features=max_words,
                stop_words='english'  # Puedes agregar stop words en español
            )
            
            ngram_matrix = vectorizer.fit_transform(texts)
            ngram_freq = ngram_matrix.sum(axis=0).A1
            ngram_names = vectorizer.get_feature_names_out()
            
            # Crear DataFrame para análisis
            ngram_df = pd.DataFrame({
                'ngram': ngram_names,
                'frequency': ngram_freq
            }).sort_values('frequency', ascending=False).head(20)
            
            # Crear visualización
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de barras
            sns.barplot(data=ngram_df, y='ngram', x='frequency', ax=ax1, palette='viridis')
            ax1.set_title(f'Top 20 {n}-gramas más frecuentes')
            ax1.set_xlabel('Frecuencia')
            
            # Gráfico de distribución
            ax2.hist(ngram_freq, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title(f'Distribución de frecuencias de {n}-gramas')
            ax2.set_xlabel('Frecuencia')
            ax2.set_ylabel('Número de n-gramas')
            
            city_name = os.path.splitext(os.path.basename(self.current_file))[0]
            fig.suptitle(f'Análisis de {n}-gramas - {city_name} ({column})', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Agregar a notebook
            self.add_plot_to_notebook(fig, f"{n}-gramas - {column}")
            
            # Mostrar estadísticas en el panel de información
            stats = f"\n🔍 ANÁLISIS DE {n}-GRAMAS:\n"
            stats += f"Total de {n}-gramas únicos: {len(ngram_names)}\n"
            stats += f"Frecuencia promedio: {np.mean(ngram_freq):.2f}\n"
            stats += f"Frecuencia máxima: {np.max(ngram_freq)}\n\n"
            stats += f"Top 10 {n}-gramas:\n"
            for i, row in ngram_df.head(10).iterrows():
                stats += f"  {row['ngram']}: {row['frequency']} veces\n"
            
            self.info_text.insert(tk.END, stats)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar n-gramas: {str(e)}")
    
    def show_stats(self):
        if not self.validate_inputs():
            return
            
        column = self.column_var.get()
        texts = self.preprocess_text(self.data[column])
        
        if not texts:
            messagebox.showwarning("Advertencia", "No hay texto válido para procesar")
            return
        
        # Calcular estadísticas
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        # Crear visualización de estadísticas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Distribución de longitud de textos (palabras)
        ax1.hist(word_counts, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_title('Distribución de longitud (palabras)')
        ax1.set_xlabel('Número de palabras')
        ax1.set_ylabel('Frecuencia')
        
        # Distribución de longitud de textos (caracteres)
        ax2.hist(char_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Distribución de longitud (caracteres)')
        ax2.set_xlabel('Número de caracteres')
        ax2.set_ylabel('Frecuencia')
        
        # Palabras más comunes
        all_words = ' '.join(texts).split()
        word_freq = Counter(all_words).most_common(15)
        words, freqs = zip(*word_freq)
        
        ax3.barh(range(len(words)), freqs, color='skyblue')
        ax3.set_yticks(range(len(words)))
        ax3.set_yticklabels(words)
        ax3.set_title('Palabras más frecuentes')
        ax3.set_xlabel('Frecuencia')
        
        # Resumen estadístico
        stats_text = f"""Estadísticas del Corpus:
        
Total de textos: {len(texts)}
Promedio palabras/texto: {np.mean(word_counts):.1f}
Promedio caracteres/texto: {np.mean(char_counts):.1f}
Total palabras únicas: {len(set(all_words))}
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax4.axis('off')
        
        city_name = os.path.splitext(os.path.basename(self.current_file))[0]
        fig.suptitle(f'Estadísticas del Corpus - {city_name} ({column})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Agregar a notebook
        self.add_plot_to_notebook(fig, f"Estadísticas - {column}")
    
    def add_plot_to_notebook(self, fig, title):
        # Crear frame para la nueva pestaña
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        
        # Agregar canvas con la figura
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Seleccionar la nueva pestaña
        self.notebook.select(frame)
    
    def clear_plots(self):
        # Limpiar todas las pestañas
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        
        # Limpiar información
        self.info_text.delete(1.0, tk.END)
        if self.data is not None:
            self.show_file_info()
    
    def validate_inputs(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero debe cargar un archivo CSV")
            return False
        
        if not self.column_var.get():
            messagebox.showwarning("Advertencia", "Seleccione una columna para analizar")
            return False
        
        try:
            n = int(self.n_var.get())
            if n < 1:
                raise ValueError("N debe ser mayor a 0")
        except ValueError:
            messagebox.showwarning("Advertencia", "N debe ser un número entero positivo")
            return False
        
        try:
            max_words = int(self.max_words_var.get())
            if max_words < 1:
                raise ValueError("Max palabras debe ser mayor a 0")
        except ValueError:
            messagebox.showwarning("Advertencia", "Max palabras debe ser un número entero positivo")
            return False
        
        return True

def main():
    root = tk.Tk()
    app = CorpusAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()