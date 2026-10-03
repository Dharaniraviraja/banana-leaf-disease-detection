import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/banana_disease_model.keras"
TEST_DIR = "dataset/split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 4

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# LOAD TEST DATA
# ==========================================

test_datagen = ImageDataGenerator(
    preprocessing_function=
    tf.keras.applications.mobilenet_v2.preprocess_input
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(test_data)

# Convert probabilities to classes
predicted_classes = (predictions >= 0.5).astype(int).flatten()

actual_classes = test_data.classes

class_names = list(test_data.class_indices.keys())

print("\n======================================")
print("       MODEL EVALUATION REPORT")
print("======================================")

print("\nClass mapping:")
print(test_data.class_indices)

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

print("\nActual vs Predicted:")

for i in range(len(actual_classes)):

    actual = class_names[actual_classes[i]]
    predicted = class_names[predicted_classes[i]]

    confidence = predictions[i][0]

    if predicted_classes[i] == 1:
        confidence = confidence
    else:
        confidence = 1 - confidence

    result = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"\nImage: {test_data.filenames[i]}"
        f"\nActual: {actual}"
        f"\nPredicted: {predicted}"
        f"\nConfidence: {confidence * 100:.2f}%"
        f"\nResult: {result}"
    )

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n======================================")
print("       CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=class_names,
        zero_division=0
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()
plt.title("Banana Leaf Disease Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved to:")
print("results/confusion_matrix.png")