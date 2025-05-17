import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import time
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
from threading import Lock

# GPU check
print("🎯 GPU-check:")
print(tf.config.list_physical_devices())
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)
        print("✅ Matrixvermenigvuldiging gedaan op GPU.")
except Exception as e:
    print("❌ GPU-test mislukt:", e)

# Folder setup
tmp = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(tmp, 'uploads')
OUTPUT_FOLDER = os.path.join(tmp, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Progress tracking
global_progress = {'total': 1, 'current': 0}
progress_lock = Lock()

@app.route('/progress')
def get_progress():
    with progress_lock:
        total = global_progress['total']
        current = global_progress['current']
    percent = int((current / total) * 100) if total > 0 else 0
    return jsonify({'value': percent})

# Load FILM model once
def load_film_model(path='film_model/saved_model'):
    print(f"[DEBUG] Loading FILM model from {path}")
    model = tf.saved_model.load(path)
    print("[DEBUG] Model loaded successfully")
    return model

film_model = load_film_model()

# Single-frame interpolation with FILM + fallback
def interpolate_single(model, a, b, t=0.5, idx=0):
    x0 = tf.expand_dims(tf.convert_to_tensor(a, tf.float32) / 255.0, 0)
    x1 = tf.expand_dims(tf.convert_to_tensor(b, tf.float32) / 255.0, 0)
    time_tensor = tf.constant([[t]], tf.float32)
    inputs = {'time': time_tensor, 'x0': x0, 'x1': x1}
    try:
        out = model(inputs, training=False)
    except Exception:
        avg = ((a.astype(np.float32) + b.astype(np.float32)) / 2.0)
        return avg.clip(0, 255).astype(np.uint8)
    frame_tensor = None
    if isinstance(out, dict):
        for val in out.values():
            if isinstance(val, tf.Tensor) and val.shape.ndims == 4:
                frame_tensor = val
                break
    elif isinstance(out, tf.Tensor) and out.shape.ndims == 4:
        frame_tensor = out
    if frame_tensor is None:
        avg = ((a.astype(np.float32) + b.astype(np.float32)) / 2.0)
        return avg.clip(0, 255).astype(np.uint8)
    frame_np = frame_tensor.numpy()[0]
    frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
    return frame_np

# Generate multiple intermediates with progress update
def interpolate_frames(model, a, b, n):
    mids = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        mid = interpolate_single(model, a, b, t, idx=i)
        mids.append(mid)
        with progress_lock:
            global_progress['current'] += 1
    return mids

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        vid = request.files.get('video')
        factor = int(request.form.get('factor', 2))
        if vid and factor in [2, 4, 8]:
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], vid.filename)
            vid.save(in_path)
            out_path = process_video(in_path, factor)
            return redirect(url_for('download', filename=os.path.basename(out_path)))
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

# Main processing: read, interpolate, and write slow-motion video with progress
def process_video(input_path, factor):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        max_width = 854
        if rgb.shape[1] > max_width:
            scale = max_width / rgb.shape[1]
            new_h = int(rgb.shape[0] * scale)
            rgb = cv2.resize(rgb, (max_width, new_h))
        frames.append(rgb)
    cap.release()
    intermediates = factor - 1
    interp_steps = (len(frames) - 1) * intermediates
    write_steps = len(frames) + interp_steps
    total_steps = interp_steps + write_steps
    with progress_lock:
        global_progress['total'] = total_steps
        global_progress['current'] = 0
    new_frames = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        new_frames.append(a)
        with progress_lock:
            global_progress['current'] += 1
        mids = interpolate_frames(film_model, a, b, intermediates)
        new_frames.extend(mids)
    new_frames.append(frames[-1])
    with progress_lock:
        global_progress['current'] += 1
    h, w = new_frames[0].shape[:2]
    out_path = os.path.join(app.config['OUTPUT_FOLDER'],
                             os.path.splitext(os.path.basename(input_path))[0] + f'_slowmo_{factor}x.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in new_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        with progress_lock:
            global_progress['current'] += 1
    writer.release()
    with progress_lock:
        global_progress['current'] = global_progress['total']
    return out_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
