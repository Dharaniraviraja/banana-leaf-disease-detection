import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. SETTINGS
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 20

TRAIN_DIR = "dataset/split/train"
TEST_DIR = "dataset/split/test"
MODEL_PATH = "model/banana_disease_model.keras"

# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    zoom_range=0.10,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:")
print(train_data.class_indices)

# ==========================================
# 3. LOAD MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pretrained layers
base_model.trainable = False

# ==========================================
# 4. BUILD MODEL
# ==========================================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

# ==========================================
# 5. COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 6. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)

# ==========================================
# 7. TRAIN
# ==========================================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# ==========================================
# 8. FINAL EVALUATION
# ==========================================

loss, accuracy = model.evaluate(test_data)

print("\n======================================")
print("       MODEL EVALUATION")
print("======================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

print("\nModel saved to:")
print(MODEL_PATH)