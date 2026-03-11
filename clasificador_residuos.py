import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========== CONFIGURACIÓN DE RUTAS ==========
# Rutas absolutas
PATH_ENTRENAMIENTO = r"c:\Users\Israr\Desktop\ESPECIALIZACIÓN DE IA Y BIG DATA\PROGRAMACIÓN DE INTELIGENCIA ARTIFICIAL\UNIDAD 2\PRACTICA 1 - CLASIFICACIÓN Y DETECCTIÓN DE RESIDUOS\dataset_reciclaje\entrenamiento"
PATH_VALIDACION = r"c:\Users\Israr\Desktop\ESPECIALIZACIÓN DE IA Y BIG DATA\PROGRAMACIÓN DE INTELIGENCIA ARTIFICIAL\UNIDAD 2\PRACTICA 1 - CLASIFICACIÓN Y DETECCTIÓN DE RESIDUOS\dataset_reciclaje\validacion"

# ========== 1. CARGAR DATASET DESDE CARPETAS ==========
print("📁 Cargando dataset...")

# Generador para datos (con augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Cargar datos de entrenamiento
train_ds = train_datagen.flow_from_directory(
    PATH_ENTRENAMIENTO,
    target_size=(224, 224),          # Tamaño para MobileNetV2
    batch_size=32,
    class_mode='binary',              # Clasificación binaria (plástico vs papel)
    classes={'plastico': 0, 'papel': 1}  # Mapeo de clases
)

# Sin dataset de validación separado - usaremos validation_split en fit()
val_ds = None

print(f"✅ Dataset cargado:")
print(f"   - Clases encontradas: {train_ds.class_indices}")
print(f"   - Total imágenes: {train_ds.samples}")

# ========== 2. FEATURE EXTRACTION CON TRANSFER LEARNING ==========
print("\n🔄 Fase 1: FEATURE EXTRACTION")
print("   Cargando MobileNetV2 preentrenado...")

# Cargar modelo preentrenado
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,                    # Sin la capa final
    weights='imagenet'                    # Pesos preentrenados
)

# Congelar el modelo base (no se entrena)
base_model.trainable = False

# Construir modelo con Feature Extraction
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),      # Reduce dimensión
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Salida binaria
])

# Compilar con learning rate NORMAL
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo construido para Feature Extraction")
print(model.summary())

# Entrenar Fase 1: Feature Extraction
print("\n🚀 Entrenando Fase 1 (Feature Extraction)...")

history_1 = model.fit(
    train_ds,
    epochs=5,
    verbose=1
)

# ========== 3. FINE-TUNING ==========
print("\n🔄 Fase 2: FINE-TUNING")
print("   Descongelando últimas capas de MobileNetV2...")

# Descongelar el modelo base
base_model.trainable = True

# Congelar las primeras capas (conservan características generales)
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompilar con learning rate MUCHO MÁS BAJO
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo listo para Fine-Tuning")

# Entrenar Fase 2: Fine-Tuning
print("\n🚀 Entrenando Fase 2 (Fine-Tuning)...")

history_2 = model.fit(
    train_ds,
    epochs=3,
    verbose=1
)

# ========== 4. COMBINACIÓN DE HISTORIALES ==========
# Combinar historiales de ambas fases
combined_history = {
    'loss': history_1.history['loss'] + history_2.history['loss'],
    'accuracy': history_1.history['accuracy'] + history_2.history['accuracy']
}

# ========== 5. EVALUACIÓN ==========
print("\n📊 EVALUACIÓN EN ENTRENAMIENTO:")
loss, accuracy = model.evaluate(train_ds)
print(f"   Pérdida: {loss:.4f}")
print(f"   Precisión: {accuracy*100:.2f}%")

# ========== 6. VISUALIZACIÓN DE RESULTADOS ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de pérdida
epochs = range(1, len(combined_history['loss']) + 1)
ax1.plot(epochs, combined_history['loss'], 'b-', linewidth=2, label='Entrenamiento')
ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5)  # Marca cambio de fase
ax1.set_title('Pérdida (Loss)')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Gráfico de precisión
ax2.plot(epochs, combined_history['accuracy'], 'b-', linewidth=2, label='Entrenamiento')
ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5)  # Marca cambio de fase
ax2.set_title('Precisión (Accuracy)')
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('entrenamiento_residuos.png')
plt.show()

# ========== 7. GUARDAR MODELO ==========
model.save('modelo_clasificador_residuos.h5')
print("\n✅ Modelo guardado como 'modelo_clasificador_residuos.h5'")

# ========== 8. PREDICCIÓN EN NUEVA IMAGEN ==========
print("\n🔍 FUNCIÓN DE PREDICCIÓN")

def predecir_residuo(ruta_imagen):
    """Predice si una imagen contiene Plástico o Papel"""
    
    # Cargar y procesar imagen
    img = tf.keras.preprocessing.image.load_img(
        ruta_imagen, 
        target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0
    
    # Predicción
    prediccion = model.predict(img_array, verbose=0)[0][0]
    
    # Interpretación
    if prediccion < 0.5:
        resultado = "PLÁSTICO"
        confianza = (1 - prediccion) * 100
    else:
        resultado = "PAPEL/CARTÓN"
        confianza = prediccion * 100
    
    return resultado, confianza

# Ejemplo de uso (descomentar cuando tengas una imagen de prueba)
# resultado, confianza = predecir_residuo("ruta/a/tu/imagen.jpg")
# print(f"Predicción: {resultado} ({confianza:.1f}% confianza)")

print("\n✅ Script completado exitosamente!")
print("\nPara hacer predicciones, usa:")
print("   resultado, confianza = predecir_residuo('ruta_a_imagen.jpg')")
print("   print(f'Predicción: {resultado} ({confianza:.1f}% confianza)')")
