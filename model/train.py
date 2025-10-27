import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Configuración
IMG_SIZE = 224  # MobileNetV2 espera 224x224
BATCH_SIZE = 32
EPOCHS = 10  # Ajusta para más precisión (e.g., 20)
NUM_CLASSES = 10
MODEL_PATH = "saved_model"  # Relativo a /model/

# Cargar y preprocesar dataset (Fashion MNIST)
print("Cargando Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizar y redimensionar (de 28x28 grayscale a 224x224 RGB)
x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0  # [0,1]
x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE))
x_test = tf.image.resize(x_test, (IMG_SIZE, IMG_SIZE))
x_train = tf.repeat(x_train, 3, axis=-1)  # Convertir a RGB
x_test = tf.repeat(x_test, 3, axis=-1)

# Labels: ya son enteros (0-9), no one-hot para sparse loss
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
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Entrenamiento
print("Entrenando modelo...")
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    callbacks=callbacks)

# Evaluación
print("Evaluando modelo...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisión en test: {test_acc:.4f}")

# Predicciones para métricas detalladas
y_pred = np.argmax(model.predict(x_test), axis=1)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))

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
model.save(full_model_path)
print(f"Modelo guardado en: model/{full_model_path}")

# Opcional: Fine-tuning (descongela y reentrena top layers)
# base_model.trainable = True
# fine_tune_at = 100  # Capas a congelar
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
# model.compile(optimizer=keras.optimizers.Adam(1e-5),  # LR bajo
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
# history_fine = model.fit(x_train, y_train, epochs=5, validation_split=0.2)
# model.save(full_model_path + "_fine_tuned")
