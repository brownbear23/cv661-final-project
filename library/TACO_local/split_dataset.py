import os
import random
import shutil
from pathlib import Path

# === CONFIGURATION ===
IMAGE_EXTS = ['.jpg', '.jpeg']
SPLITS = {
    'train': 0.75,
    'valid': 0.15,
    'test': 0.1
}
SEED = 42
# =====================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_file_stem(file_path):
    return Path(file_path).stem

def get_all_pairs(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in IMAGE_EXTS]
    paired_files = []
    for image in image_files:
        stem = get_file_stem(image)
        label_file = f"{stem}.txt"
        if os.path.exists(os.path.join(labels_dir, label_file)):
            paired_files.append(stem)
    return paired_files

def split_data(pairs, splits):
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(splits['train'] * total)
    valid_end = train_end + int(splits['valid'] * total)

    return {
        'train': pairs[:train_end],
        'valid': pairs[train_end:valid_end],
        'test': pairs[valid_end:]
    }

def copy_pairs(pairs_dict, images_dir, labels_dir, output_root):
    for split, stems in pairs_dict.items():
        img_dst = os.path.join(output_root, split, 'images')
        lbl_dst = os.path.join(output_root, split, 'labels')
        ensure_dir(img_dst)
        ensure_dir(lbl_dst)

        for stem in stems:
            # Copy image
            for ext in IMAGE_EXTS:
                src_img = os.path.join(images_dir, stem + ext)
                if os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(img_dst, stem + ext))
                    break
            # Copy label
            src_lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_dst, stem + ".txt"))

if __name__ == "__main__":
    random.seed(SEED)
    root = os.getcwd()  # Current folder (annotation_master)
    images_dir = os.path.join(root, 'images')
    labels_dir = os.path.join(root, 'labels')

    print("Gathering matching image-label pairs...")
    pairs = get_all_pairs(images_dir, labels_dir)

    print("Shuffling and splitting...")
    pairs_split = split_data(pairs, SPLITS)

    print("Creating directories and copying files...")
    copy_pairs(pairs_split, images_dir, labels_dir, root)

    print("Done! Folders 'train/', 'valid/', and 'test/' created.")
