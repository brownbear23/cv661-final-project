import os
import json
import shutil
import argparse


def coco2yolo_convert(
    json_path: str,
    data_root: str,
    output_json: str,
    images_out: str,
    labels_out: str
):
    # Load COCO annotations
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Prepare output dirs
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Build a map from image_id to its annotations
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    # Process each image entry
    for img in coco.get('images', []):
        orig_file = img['file_name']              # e.g. "batch_1/000006.jpg"
        batch, fname = orig_file.split('/', 1)    # batch = 'batch_1', fname = '000006.jpg'
        new_fname = f"{batch}_{fname}"           # batch_1_000006.jpg

        # Update JSON entry
        img['file_name'] = os.path.join(images_out, new_fname)

        # Copy image
        src = os.path.join(data_root, orig_file)
        dst = os.path.join(images_out, new_fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Image not found: {src}")
        shutil.copy2(src, dst)

        # Prepare YOLO label file
        img_w = img['width']
        img_h = img['height']
        yolo_fname = os.path.splitext(new_fname)[0] + '.txt'
        yolo_path = os.path.join(labels_out, yolo_fname)

        # Write each annotation in YOLO format: class x_center y_center width height (normalized)
        with open(yolo_path, 'w') as lbl:
            for ann in anns_by_image.get(img['id'], []):
                x, y, w, h = ann['bbox']
                x_c = x + w / 2.0
                y_c = y + h / 2.0
                # normalize
                x_norm = x_c / img_w
                y_norm = y_c / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                cls = ann['category_id']
                lbl.write(f"{cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    # Save updated JSON
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert COCO annotations with batched filenames into YOLO format and flatten images'
    )
    parser.add_argument('--json',      default='data/annotations.json', help='Path to COCO annotation JSON')
    parser.add_argument('--data_root', default='data', help='Root directory of batch folders')
    parser.add_argument('--output_json', default='annotations_yolo.json', help='Output path for updated JSON')
    parser.add_argument('--images_out',  default='images', help='Directory to store renamed images')
    parser.add_argument('--labels_out',  default='labels', help='Directory to store YOLO label files')
    args = parser.parse_args()

    coco2yolo_convert(
        json_path=args.json,
        data_root=args.data_root,
        output_json=args.output_json,
        images_out=args.images_out,
        labels_out=args.labels_out
    )
