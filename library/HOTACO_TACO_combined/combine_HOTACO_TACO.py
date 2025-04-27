import os
import shutil

# Define source directories
hotaco_base = '/workspace/cv661-final-project/library/HOTACO/data'
taco_base = '/workspace/cv661-final-project/library/TACO_local/data_splits'

# Define output directory
output_base = '/workspace/cv661-final-project/library/HOTACO_TACO_combined/data'
splits = ['train', 'valid', 'test']

# Create output structure
for split in splits:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)


# Helper to copy files
def copy_all_files(src_dir, dst_dir, prepend_name=None):
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        if os.path.isfile(src_file):
            if prepend_name:
                filename = f"{prepend_name}_{filename}"
            dst_file = os.path.join(dst_dir, filename)
            shutil.copy2(src_file, dst_file)


# Copy files from both datasets
for split in splits:
    # Copy HOTACO
    hotaco_img_dir = os.path.join(hotaco_base, split, 'images')
    hotaco_lbl_dir = os.path.join(hotaco_base, split, 'labels')
    copy_all_files(hotaco_img_dir, os.path.join(output_base, split, 'images'), prepend_name='hotaco')
    copy_all_files(hotaco_lbl_dir, os.path.join(output_base, split, 'labels'), prepend_name='hotaco')

    # Copy TACO_local
    taco_img_dir = os.path.join(taco_base, split, 'images')
    taco_lbl_dir = os.path.join(taco_base, split, 'labels')
    copy_all_files(taco_img_dir, os.path.join(output_base, split, 'images'), prepend_name='taco')
    copy_all_files(taco_lbl_dir, os.path.join(output_base, split, 'labels'), prepend_name='taco')

print("✅ All files have been copied and combined!")
