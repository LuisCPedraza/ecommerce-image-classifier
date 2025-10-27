import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Configuración optimizada para RAM baja
IMG_SIZE = 224  # MobileNetV2 espera 224x224
NUM_SAMPLES = 5000  # Submuestreo para test rápido (aumenta a 60000 después)
BATCH_SIZE = 4  # Mínimo para CPU limitada
EPOCHS = 3  # Rápido para validar
NUM_CLASSES = 10  # Fashion MNIST tiene 10 clases
SHUFFLE_BUFFER = 500  # Buffer bajo para shuffle
MODEL_PATH = "saved_model"  # Relativo a /model/

# Cargar y preprocesar dataset (Fashion MNIST) con tf.data para eficiencia
print("Cargando Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Submuestreo para eficiencia en RAM baja
indices = np.random.choice(len(x_train), min(NUM_SAMPLES, len(x_train)), replace=False)
x_train = x_train[indices]
y_train = y_train[indices]
print(f"Usando {len(x_train)} muestras de train.")

# Función de preprocesamiento (normalizar, resize, RGB)
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # [0,1]
    image = tf.expand_dims(image, -1)  # Grayscale a 1 canal
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # 224x224
    image = tf.repeat(image, 3, axis=-1)  # A RGB
    return image, label

# Datasets: Batch y prefetch para eficiencia (evita OOM)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=2)  # Bajo parallel
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(1)  # Sin cache, mínimo prefetch

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=2)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(1)

# Para validación: Split train en train/val (80/20, mínimo 1 batch)
val_size = max(1, int(0.2 * len(x_train) / BATCH_SIZE))  # Al menos 1 batch
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Augmentation para mejor generalización
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Modelo: Transfer Learning con MobileNetV2
print("Construyendo modelo...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,  # Sin clasificador final
                         weights='imagenet')  # Preentrenado
base_model.trainable = False  # Congelar base inicialmente

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Callbacks: EarlyStopping y ReduceLROnPlateau
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),  # Patience bajo para epochs=3
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)  # Patience bajo
]

# Entrenamiento
print("Entrenando modelo...")
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset,
                    callbacks=callbacks)

# Evaluación
print("Evaluando modelo...")
test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
print(f"Precisión en test: {test_acc:.4f}")

# Predicciones para métricas detalladas
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_flat = np.concatenate([y for x, y in test_dataset], axis=0)
print("\nReporte de Clasificación:")
print(classification_report(y_test_flat, y_pred, target_names=class_names))

# Plot history (opcional, guarda gráfico)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.close()

# Guardar modelo
full_model_path = os.path.join(MODEL_PATH, "ecommerce_classifier")
model.save(full_model_path + '.keras')  # Formato nativo Keras 3.x
print(f"Modelo guardado en: model/{full_model_path}")
