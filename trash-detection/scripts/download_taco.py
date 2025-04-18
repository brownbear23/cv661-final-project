# scripts/download_taco.py
'''
This script downloads TACO's images from Flickr given an annotation json file
Based on code written by Pedro F. Proenza, 2019
'''
import os
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys

def download_taco():
    # Create directories
    os.makedirs('../data/raw/taco', exist_ok=True)
    
    # First, download the annotations file if it doesn't exist
    annotations_url = "https://github.com/pedropro/TACO/raw/master/data/annotations.json"
    annotations_path = '../data/raw/taco/annotations.json'
    
    if not os.path.exists(annotations_path):
        print("Downloading TACO annotations file...")
        response = requests.get(annotations_url)
        with open(annotations_path, 'wb') as f:
            f.write(response.content)
    
    dataset_dir = '../data/raw/taco'
    
    print('Note: If for any reason the connection is broken, just run this script again and it will start where it left off.')
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.loads(f.read())
        nr_images = len(annotations['images'])
        for i in range(nr_images):
            image = annotations['images'][i]
            file_name = image['file_name']
            url_original = image['flickr_url']
            url_resized = image['flickr_640_url']
            file_path = os.path.join(dataset_dir, file_name)
            
            # Create subdir if necessary
            subdir = os.path.dirname(file_path)
            if not os.path.isdir(subdir):
                os.makedirs(subdir, exist_ok=True)
            
            if not os.path.isfile(file_path):
                try:
                    # Try original URL first
                    response = requests.get(url_original)
                    if response.status_code != 200:
                        # If original fails, try resized URL
                        response = requests.get(url_resized)
                    
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        if hasattr(img, '_getexif') and img._getexif():
                            img.save(file_path, exif=img.info["exif"])
                        else:
                            img.save(file_path)
                    else:
                        print(f"Failed to download {file_name}, status code: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading {file_name}: {str(e)}")
            
            # Show loading bar
            bar_size = 30
            x = int(bar_size * i / nr_images)
            sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (bar_size - x), i, nr_images))
            sys.stdout.flush()
        
        sys.stdout.write('Finished downloading TACO dataset\n')
        print(f"Downloaded {nr_images} images to {dataset_dir}")

if __name__ == "__main__":
    download_taco()