import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import sys
from openslide import OpenSlide


image_path = "TCGA-3C-AAAU-01A-01-TS1.2F52DD63-7476-4E85-B7C6-E06092DB6CC1.svs"

# Load phikon-v2
tokenizer = AutoImageProcessor.from_pretrained("owkin/phikon-v2",cache_dir="./model_cache")
model = AutoModel.from_pretrained("owkin/phikon-v2", cache_dir="./model_cache")

print(f"parralleizable: {model.is_parallelizable}")
print(f"base model: {model.base_model_prefix}")

# Load and preprocess image
slide = OpenSlide(image_path)
print(f"Slide dimensions: {slide.dimensions}")

# Extract a region from the slide (e.g., top-left 512x512 pixels at level 0)
region = slide.read_region((0, 0), 0, (512, 512)).convert("RGB")
# Preprocess using the processor
inputs = tokenizer(region, return_tensors="pt")

model.eval()
with torch.inference_mode():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :] 
    print(features)
    