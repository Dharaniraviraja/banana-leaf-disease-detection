from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

# --------------------------------------------------
# DATASET LOCATION
# --------------------------------------------------

dataset_folder = Path("dataset/raw/batch2")

folders = [
    ("Sigatoka", dataset_folder / "sigatoka"),
    ("Wilt", dataset_folder / "wilt")
]

image_extensions = {".jpg", ".jpeg", ".png"}

images = []


# --------------------------------------------------
# FIND LABELED IMAGES
# --------------------------------------------------

for label, folder in folders:

    print(f"\nChecking: {folder}")

    if not folder.exists():
        print("Folder not found!")
        continue

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() in image_extensions:

            images.append((image_path, label))
            print(f"Found: {image_path.name} -> {label}")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n======================================")
print("       LABELED DATASET SUMMARY")
print("======================================")

sigatoka_count = sum(
    1 for _, label in images if label == "Sigatoka"
)

wilt_count = sum(
    1 for _, label in images if label == "Wilt"
)

print(f"Sigatoka images : {sigatoka_count}")
print(f"Wilt images     : {wilt_count}")
print(f"Total images    : {len(images)}")


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

if len(images) == 0:
    print("\nNo images found.")
    exit()


columns = 4
rows = math.ceil(len(images) / columns)

fig = plt.figure(figsize=(16, rows * 4))

for i, (image_path, label) in enumerate(images):

    try:

        image = Image.open(image_path)
        image = image.convert("RGB")

        ax = plt.subplot(rows, columns, i + 1)

        ax.imshow(image)

        ax.set_title(
            f"{label}\n{image_path.name}",
            fontsize=8
        )

        ax.axis("off")

    except Exception as error:

        print(
            f"Error opening {image_path.name}: {error}"
        )


plt.tight_layout()

output_file = Path("dataset/labeled_dataset_view.png")

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print(f"\nGrid saved to: {output_file}")

plt.show()