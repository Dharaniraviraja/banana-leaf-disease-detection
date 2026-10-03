from pathlib import Path
from PIL import Image, ImageOps
import csv


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

metadata_file = Path("dataset/metadata.csv")
output_folder = Path("dataset/processed_labeled")

target_size = (224, 224)

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png"
}


# --------------------------------------------------
# CHECK METADATA
# --------------------------------------------------

if not metadata_file.exists():
    print("ERROR: metadata.csv not found.")
    exit()


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# PROCESS TRAINING IMAGES
# --------------------------------------------------

total_processed = 0
total_skipped = 0


with open(
    metadata_file,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        image_name = row["Image"]
        batch = row["Batch"]
        label = row["Label"]
        use_for_training = row["Use_For_Training"]

        # Only process labeled training images
        if label not in ["Sigatoka", "Wilt"]:
            continue

        if use_for_training != "Yes":
            continue

        # Find the original image
        image_path = (
            Path("dataset/raw")
            / batch
            / label.lower()
            / image_name
        )

        # --------------------------------------------------
        # HANDLE BATCH1 STYLE FOLDER
        # --------------------------------------------------

        if not image_path.exists():

            image_path = (
                Path("dataset/raw")
                / batch
                / image_name
            )

        # --------------------------------------------------
        # CHECK IMAGE
        # --------------------------------------------------

        if not image_path.exists():

            print(
                f"WARNING: Image not found: "
                f"{image_name}"
            )

            total_skipped += 1
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        try:

            # Open image
            image = Image.open(image_path)

            # Correct camera orientation
            image = ImageOps.exif_transpose(image)

            # Convert to RGB
            image = image.convert("RGB")

            # Preserve aspect ratio while fitting into 224x224
            image = ImageOps.pad(
                image,
                target_size,
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0)
            )

            # Create class folder
            class_folder = (
                output_folder / label
            )

            class_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            # Create safe output filename
            output_name = (
                image_path.stem
                + "_processed.jpg"
            )

            output_path = (
                class_folder / output_name
            )

            # Save
            image.save(
                output_path,
                "JPEG",
                quality=95
            )

            print(
                f"Processed: "
                f"{label}/{image_name}"
            )

            total_processed += 1

        except Exception as error:

            print(
                f"ERROR processing "
                f"{image_name}: {error}"
            )

            total_skipped += 1


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("======================================")
print("     LABELED PREPROCESSING")
print("======================================")

print(
    f"Images processed : {total_processed}"
)

print(
    f"Images skipped   : {total_skipped}"
)

print()
print(
    f"Output folder: {output_folder}"
)

print()
print("Preprocessing completed!")