import requests
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer, pipeline
from datasets import load_dataset

