from pathlib import Path
import csv

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

raw_folder = Path("dataset/raw")
output_file = Path("dataset/metadata.csv")

image_extensions = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------
# GPS IMAGES
# --------------------------------------------------

gps_images = {
    "projectimage2.jpeg",

    "WhatsApp Image 2026-09-25 at 7.26.07 PM.jpeg",
    "WhatsApp Image 2026-09-25 at 7.26.08 PM.jpeg"
}


# --------------------------------------------------
# CREATE METADATA
# --------------------------------------------------

rows = []

for batch_folder in sorted(raw_folder.iterdir()):

    if not batch_folder.is_dir():
        continue

    batch_name = batch_folder.name

    # Check for disease folders
    disease_folders = [
        folder for folder in batch_folder.iterdir()
        if folder.is_dir()
    ]

    # --------------------------------------------------
    # BATCH WITH DISEASE FOLDERS
    # --------------------------------------------------

    if disease_folders:

        for disease_folder in sorted(disease_folders):

            folder_name = disease_folder.name.lower()

            if folder_name == "sigatoka":
                label = "Sigatoka"

            elif folder_name == "wilt":
                label = "Wilt"

            else:
                label = disease_folder.name

            for image_path in sorted(disease_folder.iterdir()):

                if image_path.suffix.lower() not in image_extensions:
                    continue

                if image_path.name in gps_images:

                    quality = "Review"
                    use_for_training = "No"
                    notes = "GPS Map Camera overlay"

                else:

                    quality = "Review"
                    use_for_training = "Yes"
                    notes = ""

                rows.append([
                    image_path.name,
                    batch_name,
                    label,
                    "Field",
                    quality,
                    use_for_training,
                    notes
                ])

    # --------------------------------------------------
    # BATCH WITHOUT DISEASE FOLDERS
    # --------------------------------------------------

    else:

        for image_path in sorted(batch_folder.iterdir()):

            if image_path.suffix.lower() not in image_extensions:
                continue

            label = "Unknown"
            quality = "Review"
            use_for_training = "Review"
            notes = ""

            if image_path.name in gps_images:

                use_for_training = "No"
                notes = "GPS Map Camera overlay"

            rows.append([
                image_path.name,
                batch_name,
                label,
                "Field",
                quality,
                use_for_training,
                notes
            ])


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Batch",
        "Label",
        "Source",
        "Quality",
        "Use_For_Training",
        "Notes"
    ])

    writer.writerows(rows)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

sigatoka_count = sum(
    1 for row in rows
    if row[2] == "Sigatoka"
)

wilt_count = sum(
    1 for row in rows
    if row[2] == "Wilt"
)

unknown_count = sum(
    1 for row in rows
    if row[2] == "Unknown"
)

training_count = sum(
    1 for row in rows
    if row[5] == "Yes"
)

excluded_count = sum(
    1 for row in rows
    if row[5] == "No"
)


print()
print("======================================")
print("       FINAL DATASET METADATA")
print("======================================")

print(f"Total images       : {len(rows)}")
print(f"Sigatoka           : {sigatoka_count}")
print(f"Wilt               : {wilt_count}")
print(f"Unknown            : {unknown_count}")
print(f"Training candidates: {training_count}")
print(f"Excluded           : {excluded_count}")

print()
print(f"Saved to: {output_file}")

print()
print("GPS images excluded from training:")

for image in gps_images:
    print(f"  - {image}")

print()
print("Metadata creation completed!")