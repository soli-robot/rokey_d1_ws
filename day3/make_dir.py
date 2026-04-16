import os
import shutil
import random

SOURCE_DIR = "/home/rokey/amr_capture"
DATASET_DIR = "my_data"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directory ready: {path}")


def clear_directory(path):
    if not os.path.exists(path):
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted directory: {item_path}")
        else:
            os.remove(item_path)
            print(f"Deleted file: {item_path}")


def prepare_dataset_structure():
    create_directory(DATASET_DIR)

    train_images = os.path.join(DATASET_DIR, "train", "images")
    train_labels = os.path.join(DATASET_DIR, "train", "labels")

    valid_images = os.path.join(DATASET_DIR, "valid", "images")
    valid_labels = os.path.join(DATASET_DIR, "valid", "labels")

    test_images = os.path.join(DATASET_DIR, "test", "images")
    test_labels = os.path.join(DATASET_DIR, "test", "labels")

    create_directory(train_images)
    create_directory(train_labels)
    create_directory(valid_images)
    create_directory(valid_labels)
    create_directory(test_images)
    create_directory(test_labels)

    clear_directory(train_images)
    clear_directory(train_labels)
    clear_directory(valid_images)
    clear_directory(valid_labels)
    clear_directory(test_images)
    clear_directory(test_labels)

    return train_images, valid_images, test_images


def get_image_files(source_dir):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = []
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(IMAGE_EXTENSIONS):
            files.append(file_name)

    return files


def split_files(files):
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_files = files[:train_end]
    valid_files = files[train_end:valid_end]
    test_files = files[valid_end:]

    return train_files, valid_files, test_files


def copy_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(target_dir, file_name)
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")


def main():
    train_dir, valid_dir, test_dir = prepare_dataset_structure()

    image_files = get_image_files(SOURCE_DIR)

    if not image_files:
        print("No image files found.")
        return

    train_files, valid_files, test_files = split_files(image_files)

    copy_files(train_files, SOURCE_DIR, train_dir)
    copy_files(valid_files, SOURCE_DIR, valid_dir)
    copy_files(test_files, SOURCE_DIR, test_dir)

    print("\nDone.")
    print(f"Total images: {len(image_files)}")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(valid_files)}")
    print(f"Test : {len(test_files)}")


if __name__ == "__main__":
    main()