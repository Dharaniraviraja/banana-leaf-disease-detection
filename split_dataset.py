from pathlib import Path
import shutil
import random

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

source_folder = Path("dataset/processed_labeled")
split_folder = Path("dataset/split")

classes = ["Sigatoka", "Wilt"]

# Fixed seed so the same split is produced every time
random.seed(42)

# Number of test images for each class
test_count = {
    "Sigatoka": 2,
    "Wilt": 1
}

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png"
}


# --------------------------------------------------
# CHECK SOURCE
# --------------------------------------------------

if not source_folder.exists():
    print("ERROR: processed_labeled folder not found.")
    exit()


# --------------------------------------------------
# REMOVE OLD SPLIT
# --------------------------------------------------

if split_folder.exists():

    print("Removing previous split...")

    shutil.rmtree(split_folder)


# --------------------------------------------------
# CREATE FOLDERS
# --------------------------------------------------

for split_name in ["train", "test"]:

    for class_name in classes:

        folder = (
            split_folder
            / split_name
            / class_name
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# --------------------------------------------------
# SPLIT EACH CLASS
# --------------------------------------------------

total_train = 0
total_test = 0


for class_name in classes:

    class_folder = source_folder / class_name

    if not class_folder.exists():

        print(
            f"ERROR: {class_name} folder not found."
        )

        continue

    images = [
        image for image in class_folder.iterdir()
        if image.suffix.lower() in image_extensions
    ]

    # Sort first so the selection is reproducible
    images = sorted(images)

    # Shuffle using fixed random seed
    random.shuffle(images)

    required_test = test_count[class_name]

    if len(images) <= required_test:

        print(
            f"ERROR: Not enough images for "
            f"{class_name}."
        )

        continue

    test_images = images[:required_test]
    train_images = images[required_test:]

    # --------------------------------------------------
    # COPY TRAIN IMAGES
    # --------------------------------------------------

    for image in train_images:

        destination = (
            split_folder
            / "train"
            / class_name
            / image.name
        )

        shutil.copy2(
            image,
            destination
        )

        total_train += 1


    # --------------------------------------------------
    # COPY TEST IMAGES
    # --------------------------------------------------

    for image in test_images:

        destination = (
            split_folder
            / "test"
            / class_name
            / image.name
        )

        shutil.copy2(
            image,
            destination
        )

        total_test += 1


    # --------------------------------------------------
    # PRINT CLASS SUMMARY
    # --------------------------------------------------

    print()
    print(class_name)
    print("----------------------")
    print(f"Total : {len(images)}")
    print(f"Train : {len(train_images)}")
    print(f"Test  : {len(test_images)}")


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print()
print("======================================")
print("        DATASET SPLIT COMPLETE")
print("======================================")

print(f"Training images : {total_train}")
print(f"Testing images  : {total_test}")
print(f"Total images    : {total_train + total_test}")

print()
print("Dataset saved to:")
print(split_folder)