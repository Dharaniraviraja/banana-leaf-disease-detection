from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

processed_folder = Path("dataset/processed_labeled")

classes = [
    "Sigatoka",
    "Wilt"
]

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png"
}

images = []


# --------------------------------------------------
# FIND PROCESSED IMAGES
# --------------------------------------------------

for label in classes:

    folder = processed_folder / label

    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() in image_extensions:
            images.append((image_path, label))


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("======================================")
print("     PROCESSED DATASET VIEWER")
print("======================================")

sigatoka_count = sum(
    1 for _, label in images
    if label == "Sigatoka"
)

wilt_count = sum(
    1 for _, label in images
    if label == "Wilt"
)

print(f"Sigatoka : {sigatoka_count}")
print(f"Wilt     : {wilt_count}")
print(f"Total    : {len(images)}")


# --------------------------------------------------
# DISPLAY GRID
# --------------------------------------------------

columns = 4
rows = math.ceil(len(images) / columns)

plt.figure(figsize=(16, rows * 4))

for i, (image_path, label) in enumerate(images):

    image = Image.open(image_path)

    plt.subplot(rows, columns, i + 1)

    plt.imshow(image)

    plt.title(
        f"{label}\n{image_path.name}",
        fontsize=8
    )

    plt.axis("off")


plt.tight_layout()

output_file = Path(
    "dataset/processed_labeled_view.png"
)

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print()
print(f"Saved to: {output_file}")
print("Processed dataset visualization completed!")