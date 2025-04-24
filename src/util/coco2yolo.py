# Modified from https://pypi.org/project/COCO-to-YOLO/

import argparse
from tqdm import tqdm
import random
import yaml
import json
import sys
import os
import shutil
import glob


def has_valid_imagedir(input_dir):
    print(f"[DEBUG] Searching for images under: {input_dir}/**/")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.bmp', '*.png']:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    print(f"[DEBUG] Found {len(image_files)} image(s)")
    return len(image_files) > 0


def has_valid_annotationdir(input_dir):
    annotations_path = f'{input_dir}/annotations'
    if not os.path.isdir(annotations_path):
        return False
    if len(glob.glob(f'{annotations_path}/*.json')) != 1:
        return False
    return True


def validate_input(input_dir):
    if not has_valid_imagedir(input_dir):
        print("Please provide a valid image directory with at least one input image (jpg, jpeg, bmp, png).")
        sys.exit(1)
    if not has_valid_annotationdir(input_dir):
        print("Please provide a valid annotations directory with exactly one COCO annotations file (json).")
        sys.exit(1)


def create_yolo_structure(output_dir, name, test_ratio, val_ratio):
    os.makedirs(f'{output_dir}/{name}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/labels/train', exist_ok=True)
    if test_ratio > 0.0:
        os.makedirs(f'{output_dir}/{name}/images/test_coco_dir', exist_ok=True)
        os.makedirs(f'{output_dir}/{name}/labels/test_coco_dir', exist_ok=True)
    if val_ratio > 0.0:
        os.makedirs(f'{output_dir}/{name}/images/valid', exist_ok=True)
        os.makedirs(f'{output_dir}/{name}/labels/valid', exist_ok=True)


def create_yaml(output_dir, name, test_ratio, val_ratio, classes):
    dataset_dict = {
        'path': '.',
        'train': f'{name}/images/train'
    }
    if test_ratio > 0.0 and val_ratio > 0.0:
        dataset_dict['test'] = f'{name}/images/test_coco_dir'
        dataset_dict['val'] = f'{name}/images/valid'
    if test_ratio > 0.0 and val_ratio == 0.0:
        dataset_dict['val'] = f'{name}/images/test_coco_dir'

    class_dict = {i: cls for i, cls in enumerate(classes)}
    dataset_dict['names'] = class_dict

    with open(f'{output_dir}/{name}.yaml', 'w') as f:
        yaml.dump(dataset_dict, f)


def create_splits(ids, test_ratio, val_ratio):
    n_test = int(test_ratio * len(ids))
    n_val = int(val_ratio * len(ids))
    test_ids = random.sample(ids, n_test)
    remaining_ids = [id for id in ids if id not in test_ids]
    val_ids = random.sample(remaining_ids, n_val)
    train_ids = [id for id in remaining_ids if id not in val_ids]
    return train_ids, test_ids, val_ids


def copy_images(input_dir, output_dir, name, ids, split, img_id_map):
    print(f'Copying {split} images...')
    for id in tqdm(ids):
        fn = img_id_map[id]  # e.g., "batch_1/000055.jpg"
        src_path = os.path.join(input_dir, fn)
        dst_path = os.path.join(output_dir, name, 'images', split, os.path.basename(fn))

        if not os.path.exists(src_path):
            print(f"[WARNING] Missing image: {src_path}")
            continue

        shutil.copyfile(src_path, dst_path)


def to_yolo_bbox(bbox, im_w, im_h):
    x, y, w, h = bbox
    return [
        (x + w / 2) / im_w,
        (y + h / 2) / im_h,
        w / im_w,
        h / im_h
    ]


def create_annotation_map(coco):
    img_id_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_id_to_anns.setdefault(img_id, []).append(ann)
    return img_id_to_anns


def create_annotations(output_dir, name, split, coco, img_ids):
    img_id_to_anns = create_annotation_map(coco)
    img_id_to_imgs = {img['id']: img for img in coco['images']}

    print(f'Creating {split} labels...')
    for img_id in tqdm(img_ids):
        img = img_id_to_imgs[img_id]
        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            continue

        width, height = img['width'], img['height']
        lines = []

        for ann in anns:
            cat_id = ann['category_id'] - 1  # assumes category_id starts from 1
            bbox = to_yolo_bbox(ann['bbox'], width, height)
            lines.append(f"{cat_id} {' '.join(f'{x:.6f}' for x in bbox)}")

        image_id = os.path.splitext(os.path.basename(img['file_name']))[0]
        label_path = os.path.join(output_dir, name, 'labels', split, f"{image_id}.txt")

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))


def convert(args):
    annotations_path = glob.glob(f'{args.input_dir}/annotations/*.json')[0]
    with open(annotations_path) as f:
        coco = json.load(f)

    classes = [cat['name'] for cat in coco['categories']]
    create_yaml(args.output_dir, args.dataset_name, args.test_ratio, args.val_ratio, classes)

    img_ids = [img['id'] for img in coco['images']]
    splits = create_splits(img_ids, args.test_ratio, args.val_ratio)
    img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    for ids, split in zip(splits, ['train', 'test_coco_dir', 'valid']):
        copy_images(args.input_dir, args.output_dir, args.dataset_name, ids, split, img_id_to_file)
        create_annotations(args.output_dir, args.dataset_name, split, coco, ids)


def run_coco_to_yolo(input_dir, output_dir, dataset_name="converted", test_ratio=0.1, val_ratio=0.0):
    class Args:
        def __init__(self):
            self.input_dir = input_dir
            self.output_dir = output_dir
            self.dataset_name = dataset_name
            self.test_ratio = test_ratio
            self.val_ratio = val_ratio

    args = Args()
    validate_input(args.input_dir)
    create_yolo_structure(args.output_dir, args.dataset_name, args.test_ratio, args.val_ratio)
    convert(args)


if __name__ == '__main__':
    run_coco_to_yolo(
        input_dir="/Users/billhan/Desktop/Dev-JHU25SS/CV/cv-final-project/src/util/test_coco_dir",
        output_dir="/Users/billhan/Desktop/Dev-JHU25SS/CV/cv-final-project/src/util/test_output",
        dataset_name="my_dataset",
        test_ratio=0.1,
        val_ratio=0.1
    )
