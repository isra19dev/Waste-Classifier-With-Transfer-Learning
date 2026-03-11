import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from pathlib import Path

# ========== CONFIGURACIÓN DE PÁGINA ==========
st.set_page_config(
    page_title="Clasificador de Residuos",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ESTILOS CSS ==========
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: bold;
    }
    .resultado-plastico {
        background-color: #FFE5E5;
        border-left: 4px solid #FF6B6B;
        padding: 20px;
        border-radius: 5px;
        color: #C92A2A;
    }
    .resultado-papel {
        background-color: #E5F5E5;
        border-left: 4px solid #51CF66;
        padding: 20px;
        border-radius: 5px;
        color: #2B8A3E;
    }
    .confianza-alta {
        color: #51CF66;
        font-weight: bold;
        font-size: 24px;
    }
    .confianza-media {
        color: #FFD43B;
        font-weight: bold;
        font-size: 24px;
    }
    .confianza-baja {
        color: #FF8787;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

#? ========== CARGAR MODELO ==========
@st.cache_resource
def cargar_modelo():
    ruta_modelo = r"c:\Users\Israr\Desktop\ESPECIALIZACIÓN DE IA Y BIG DATA\PROGRAMACIÓN DE INTELIGENCIA ARTIFICIAL\UNIDAD 2\PRACTICA 1 - CLASIFICACIÓN Y DETECCTIÓN DE RESIDUOS\modelo_clasificador_residuos.h5"
    return tf.keras.models.load_model(ruta_modelo)

#? ========== FUNCIÓN DE PREDICCIÓN ==========
def predecir_imagen(imagen, model):
    """Predice si una imagen es Plástico o Papel"""
    
    #! Redimensionar imagen
    img_resized = imagen.resize((224, 224))
    
    #! Convertir a array y normalizar
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    #! Hacer predicción
    prediccion = model.predict(img_array, verbose=0)[0][0]
    
    #! Clasificar
    if prediccion < 0.5:
        clase = "PLÁSTICO"
        emoji = "🟦"
        confianza = float((1 - prediccion) * 100)
    else:
        clase = "PAPEL/CARTÓN"
        emoji = "📄"
        confianza = float(prediccion * 100)
    
    return clase, emoji, confianza, float(prediccion)

#? ========== INTERFAZ PRINCIPAL ==========
col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://img.icons8.com/color/96/000000/trash.png", width=80)

with col2:
    st.title("♻️ Clasificador de Residuos")
    st.markdown("**Plástico vs Papel/Cartón** - con Transfer Learning")

st.markdown("---")

# Cargar modelo
with st.spinner("⏳ Cargando modelo..."):
    model = cargar_modelo()
st.success("✅ Modelo cargado correctamente")

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["📸 Subir Imagen", "📁 Usar Dataset", "ℹ️ Información"])

# ========== TAB 1: SUBIR IMAGEN ==========
with tab1:
    st.header("Sube una imagen para clasificarla")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        archivo_subido = st.file_uploader(
            "Elige una imagen",
            type=["jpg", "jpeg", "png", "bmp"],
            help="JPG, PNG, BMP - máximo 200MB"
        )
    
    with col2:
        st.info("""
        💡 **Tips:**
        - Buena iluminación
        - Residuo limpio
        - Imagen clara y nítida
        """)
    
    if archivo_subido is not None:
        # Mostrar imagen
        imagen = Image.open(archivo_subido)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Imagen original")
            st.image(imagen, use_column_width=True)
        
        with col2:
            st.markdown("### 🤖 Predicción")
            
            # Hacer predicción
            clase, emoji, confianza, valor_bruto = predecir_imagen(imagen, model)
            
            # Mostrar resultado según clase
            if clase == "PLÁSTICO":
                st.markdown(f"""
                    <div class="resultado-plastico">
                    <h2>{emoji} {clase}</h2>
                    <p>Confianza: <span class="confianza-alta">{confianza:.2f}%</span></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="resultado-papel">
                    <h2>{emoji} {clase}</h2>
                    <p>Confianza: <span class="confianza-alta">{confianza:.2f}%</span></p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Barra de confianza
            st.progress(confianza / 100)
            
            # Detalles técnicos
            with st.expander("📊 Detalles técnicos"):
                st.write(f"Valor bruto de salida: {valor_bruto:.4f}")
                st.write(f"Umbral de decisión: 0.5000")
                if valor_bruto < 0.5:
                    st.write(f"Diferencia del umbral: {abs(valor_bruto - 0.5):.4f} (Hacia PLÁSTICO)")
                else:
                    st.write(f"Diferencia del umbral: {abs(valor_bruto - 0.5):.4f} (Hacia PAPEL)")

# ========== TAB 2: DATASET ==========
with tab2:
    st.header("🗂️ Pruebas con imágenes del dataset")
    
    dataset_path = r"c:\Users\Israr\Desktop\ESPECIALIZACIÓN DE IA Y BIG DATA\PROGRAMACIÓN DE INTELIGENCIA ARTIFICIAL\UNIDAD 2\PRACTICA 1 - CLASIFICACIÓN Y DETECCTIÓN DE RESIDUOS\dataset_reciclaje"
    
    if os.path.exists(dataset_path):
        # Buscar imágenes en el dataset
        imagenes_disponibles = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    imagenes_disponibles.append(os.path.join(root, file))
        
        if imagenes_disponibles:
            st.markdown(f"**Imágenes encontradas:** {len(imagenes_disponibles)}")
            
            # Selector de imagen
            imagen_seleccionada = st.selectbox(
                "Elige una imagen",
                imagenes_disponibles,
                format_func=lambda x: os.path.basename(x)
            )
            
            if imagen_seleccionada:
                imagen = Image.open(imagen_seleccionada)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📷 Imagen")
                    st.image(imagen, use_column_width=True)
                
                with col2:
                    st.markdown("### 🤖 Predicción")
                    
                    clase, emoji, confianza, valor_bruto = predecir_imagen(imagen, model)
                    
                    if clase == "PLÁSTICO":
                        st.markdown(f"""
                            <div class="resultado-plastico">
                            <h2>{emoji} {clase}</h2>
                            <p>Confianza: <span class="confianza-alta">{confianza:.2f}%</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="resultado-papel">
                            <h2>{emoji} {clase}</h2>
                            <p>Confianza: <span class="confianza-alta">{confianza:.2f}%</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.progress(confianza / 100)
                    
                    st.markdown(f"**Archivo:** `{os.path.basename(imagen_seleccionada)}`")
        else:
            st.warning("⚠️ No hay imágenes en el dataset")
    else:
        st.error(f"❌ Dataset no encontrado en: {dataset_path}")

# ========== TAB 3: INFORMACIÓN ==========
with tab3:
    st.header("ℹ️ Sobre este modelo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Características del modelo")
        st.write("""
        - **Arquitectura:** MobileNetV2 + Transfer Learning
        - **Datos de entrenamiento:** 66 imágenes
        - **Clases:** Plástico (0) y Papel/Cartón (1)
        - **Precisión:** 98.48%
        - **Métodos:** Feature Extraction + Fine-Tuning
        """)
    
    with col2:
        st.subheader("🎯 Cómo funciona")
        st.write("""
        1. **Feature Extraction:** Se usan características preentrenadas de ImageNet
        2. **Fine-Tuning:** Se ajustan las últimas capas específicamente para residuos
        3. **Predicción:** La imagen se procesa y se obtiene probabilidad de cada clase
        4. **Clasificación:** Se asigna a plástico o papel según el umbral 0.5
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Fases de entrenamiento")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### Fase 1: Feature Extraction
        - **Epocas:** 5
        - **Learning Rate:** 1e-3
        - **Frozen Layers:** Sí
        """)
    
    with col2:
        st.markdown("""
        #### Fase 2: Fine-Tuning
        - **Epocas:** 3
        - **Learning Rate:** 1e-5
        - **Últimas capas:** Desentrenadas
        """)
    
    # Mostrar gráfica de entrenamiento si existe
    grafica_path = r"c:\Users\Israr\Desktop\ESPECIALIZACIÓN DE IA Y BIG DATA\PROGRAMACIÓN DE INTELIGENCIA ARTIFICIAL\UNIDAD 2\PRACTICA 1 - CLASIFICACIÓN Y DETECCTIÓN DE RESIDUOS\entrenamiento_residuos.png"
    if os.path.exists(grafica_path):
        st.markdown("---")
        st.subheader("📊 Gráficas de entrenamiento")
        st.image(grafica_path, use_column_width=True)
    
    st.markdown("---")
    st.info("""
    💡 **Nota:** Este modelo fue entrenado con Transfer Learning usando MobileNetV2, 
    que fue previamente entrenada en ImageNet (contiene millones de imágenes). 
    Esto permite aprender patrones complejos con pocas imágenes de entrenamiento.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚀 Clasificador de Residuos © 2026 | Transfer Learning with TensorFlow</p>
</div>
""", unsafe_allow_html=True)
