import tensorflow_hub as hub
import tensorflow as tf
import os

HUB_URL = "https://tfhub.dev/google/film/1"
OUT_DIR = "film_model/saved_model"

print("Loading FILM model van TF-Hub…")
model = hub.load(HUB_URL)

print(f"Saving naar {OUT_DIR} …")
tf.saved_model.save(model, OUT_DIR)

print("Klaar!")
