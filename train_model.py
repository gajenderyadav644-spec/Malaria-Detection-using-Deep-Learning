import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================
# Dataset Path
# =========================
data_dir = "cell_images"

img_size = 64
batch_size = 32

# =========================
# Load Dataset
# =========================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

# =========================
# Normalize Images
# =========================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# =========================
# Build CNN Model
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# =========================
# Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Train Model
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# =========================
# Save Model
# =========================
model.save("malaria_model.keras")

print("\n✅ Training Complete!")
print("Model saved as malaria_model.keras")