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

# Load FILM model once
def load_film_model(path='film_model/saved_model'):
    print(f"[DEBUG] Loading FILM model from {path}")
    model = tf.saved_model.load(path)
    print("[DEBUG] Model loaded successfully")
    return model

film_model = load_film_model()

# Single-frame interpolation with FILM + fallback on error or invalid output
def interpolate_single(model, a, b, t=0.5, idx=0):
    print(f"[DEBUG] interpolate_single idx={idx}, t={t}")
    # Convert to float32 [0,1]
    x0 = tf.expand_dims(tf.convert_to_tensor(a, tf.float32) / 255.0, 0)
    x1 = tf.expand_dims(tf.convert_to_tensor(b, tf.float32) / 255.0, 0)
    time_tensor = tf.constant([[t]], tf.float32)
    inputs = {'time': time_tensor, 'x0': x0, 'x1': x1}
    try:
        out = model(inputs, training=False)
    except Exception as e:
        print(f"[ERROR] Model call failed at idx={idx}, t={t}: {e} - using fallback average")
        avg = ((a.astype(np.float32) + b.astype(np.float32)) / 2.0)
        return avg.clip(0, 255).astype(np.uint8)

    # Select the image tensor from outputs\    
    frame_tensor = None
    if isinstance(out, dict):
        for key, val in out.items():
            if isinstance(val, tf.Tensor) and val.shape.ndims == 4:
                frame_tensor = val
                print(f"[DEBUG] Selected tensor key: {key}")
                break
    elif isinstance(out, tf.Tensor) and out.shape.ndims == 4:
        frame_tensor = out
    if frame_tensor is None:
        print(f"[ERROR] No valid tensor found in model output at idx={idx}, t={t} - using fallback average")
        avg = ((a.astype(np.float32) + b.astype(np.float32)) / 2.0)
        return avg.clip(0, 255).astype(np.uint8)

    # Convert tensor to numpy and denormalize
    try:
        frame_np = frame_tensor.numpy()[0]
        frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
        # Validate output
        if not np.isfinite(frame_np).all():
            raise ValueError("Non-finite values in frame")
        return frame_np
    except Exception as e:
        print(f"[ERROR] Processing output failed at idx={idx}, t={t}: {e} - using fallback average")
        avg = ((a.astype(np.float32) + b.astype(np.float32)) / 2.0)
        return avg.clip(0, 255).astype(np.uint8)

# Generate multiple intermediates between two frames
def interpolate_frames(model, a, b, n):
    print(f"[DEBUG] interpolate_frames called n={n}")
    mids = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        print(f"[DEBUG] Generating interp {i}/{n} at t={t}")
        mid = interpolate_single(model, a, b, t, idx=i)
        mids.append(mid)
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

# Main processing: read, interpolate, and write slow-motion video
def process_video(input_path, factor):
    print(f"[INFO] process_video factor={factor}, input={input_path}")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR->RGB for model
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames")

    intermediates = factor - 1
    new_frames = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        new_frames.append(a)
        mids = interpolate_frames(film_model, a, b, intermediates)
        new_frames.extend(mids)
    new_frames.append(frames[-1])
    print(f"[INFO] Total output frames: {len(new_frames)}")

    # Write with original FPS for slow motion
    h, w = new_frames[0].shape[:2]
    out_name = os.path.splitext(os.path.basename(input_path))[0] + f'_slowmo_{factor}x.mp4'
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )
    for idx, frame in enumerate(new_frames, 1):
        # Convert RGB->BGR for writing
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        if idx % 50 == 0:
            print(f"[DEBUG] Written {idx}/{len(new_frames)}")
    writer.release()
    print(f"[INFO] Saved output to {out_path}")
    return out_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)