from transformers import AutoModel, AutoFeatureExtractor

model = AutoModel.from_pretrained("owkin/phikon-v2")
feature_extractor = AutoFeatureExtractor.from_pretrained("owkin/phikon-v2")