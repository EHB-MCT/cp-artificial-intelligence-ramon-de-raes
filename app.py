import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import time
from flask import Flask, render_template, request, send_file, redirect, url_for

# Folder setup
tmp = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(tmp, 'uploads')
OUTPUT_FOLDER = os.path.join(tmp, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the FILM model (custom logic may vary depending on how the model is saved)
def load_film_model(model_path='film_model/saved_model'):
    print("[DEBUG] Loading model...")
    try:
        model = tf.saved_model.load(model_path)
        print("[DEBUG] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None

# Interpolate frames: given two frames, generate one in-between
def interpolate_frame(model, frame1, frame2):
    print("[DEBUG] Begin interpolatie van een framepaar")

    # Verwijder de conversie naar tensor als een test zonder model
    x0 = frame1.astype(np.float32)  # Directe numpy array
    x1 = frame2.astype(np.float32)  # Directe numpy array

    print(f"[DEBUG] Shape van x0: {x0.shape}, x1: {x1.shape}")

    # Testen zonder het model (dupliceren frames voor eenvoud)
    mid_frame = (x0 + x1) // 2  # Gemiddelde van de twee frames

    print("[DEBUG] Test interpolatie zonder model. Gebruik gemiddelde van frames.")
    return mid_frame.astype(np.uint8)

# Functie voor het instellen van een timeout
def run_with_timeout(model, inputs, timeout=60):
    start_time = time.time()
    try:
        result = model(inputs, training=False)
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Model execution took too long: {elapsed_time:.2f}s")
        print(f"[DEBUG] Model executed in {elapsed_time:.2f}s")
        return result
    except Exception as e:
        print(f"[ERROR] Timeout of fout bij modeluitvoering: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        vid = request.files.get('video')
        if vid:
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], vid.filename)
            vid.save(in_path)
            out_path = process_video(in_path)
            return redirect(url_for('download', filename=os.path.basename(out_path)))
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(path, as_attachment=True)

# Main processing: split frames, interpolate, reassemble
def process_video(input_path):
    print("[INFO] Start video processing")
    start_total = time.time()

    print("[INFO] Loading FILM model...")
    model = load_film_model()

    print(f"[INFO] Reading video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video file: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames.")

    print("[INFO] Interpolating frames...")
    new_frames = []
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        try:
            mid = interpolate_frame(model, f1, f2)
            new_frames.extend([f1, mid])
        except Exception as e:
            print(f"[ERROR] Failed to interpolate frame pair {i}-{i+1}: {e}")
            new_frames.extend([f1, f2])  # fallback: no interpolation

    new_frames.append(frames[-1])

    height, width, _ = new_frames[0].shape
    out_filename = os.path.splitext(os.path.basename(input_path))[0] + '_interp.mp4'
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

    print(f"[INFO] Writing interpolated video to: {out_path}")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps * 2, (width, height))
    for frame in new_frames:
        writer.write(frame)
    writer.release()

    total_time = time.time() - start_total
    print(f"[INFO] Done. Total processing time: {total_time:.2f} seconds")
    return out_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
