# HOpkins Trash Annotations in COntext (HOTACO)

## Getting Started

### Download TACO dataset
Locate to `trash-detection/scripts` to run the following file to `data/raw/taco`. This might take ~15 minutes to run.

``` bash
python3 download_taco.py
```

### RunPod setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir /workspace/lib/
mkdir /workspace/lib/PY

echo 'export PIP_CACHE_DIR=/workspace/lib/PY' >> ~/.bashrc
source ~/.bashrc

source .venv/bin/activate

echo $PIP_CACHE_DIR

git submodule init
git submodule update --recursive

```

## Modules trained
**Train Model #1: Fine-Tuning after Transfer Learning**

Fine-tune a checkpoint model (`best.pt`) that was previously transfer-learned on the TACO dataset. Use [`yolov8m.pt`](http://yolov8m.pt) as the original base model, and fine-tune it on the HOTACO dataset to specialize it for your target domain.

(Fine-tuning: further training a model that has already undergone transfer learning, typically with a lower learning rate, to adapt it more precisely to a new dataset.)
    

**Train Model #2: Single-Stage Transfer Learning**
    
Use [`yolov11m.pt`](http://yolov11m.pt/) as a base model pretrained on a general dataset (e.g., COCO), and perform one-shot transfer learning directly on the HOTACO dataset.
    
(Transfer learning: adapting a pretrained model to a new task or dataset, typically by training some or all layers on new data.)
    

**Train Model #3: Sequential Transfer Learning**
    
Use [`yolov11m.pt`](http://yolov11m.pt/) as the base model. First, perform transfer learning on the TACO dataset, then fine-tune the resulting model on the HOTACO dataset.
    
(Sequential transfer learning: performing transfer learning in multiple stages, progressively adapting the model from general to domain-specific datasets.)