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

# Load the FILM model
def load_film_model(model_path='film_model/saved_model'):
    print("[DEBUG] Loading FILM model...")
    try:
        model = tf.saved_model.load(model_path)
        print("[DEBUG] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

# Use FILM to interpolate between two frames
def interpolate_frames(model, f1, f2, num_intermediate):
    # Normaliseer inputframes naar [0.0, 1.0]
    image0 = tf.convert_to_tensor(f1[np.newaxis, ...] / 255.0, dtype=tf.float32)
    image1 = tf.convert_to_tensor(f2[np.newaxis, ...] / 255.0, dtype=tf.float32)

    times = [(i + 1) / (num_intermediate + 1) for i in range(num_intermediate)]
    interpolated_frames = []

    for t in times:
        t_tensor = tf.convert_to_tensor([[t]], dtype=tf.float32)
        try:
            prediction = model({'x0': image0, 'x1': image1, 'time': t_tensor}, training=False)
            interpolated_tensor = prediction['image'] if isinstance(prediction, dict) else prediction
            interpolated_np = interpolated_tensor.numpy()[0]  # shape: (H, W, 3)

            # Denormaliseer naar [0–255] en zet om naar uint8
            interpolated_np = np.clip(interpolated_np * 255.0, 0, 255).astype(np.uint8)
            interpolated_frames.append(interpolated_np)

        except Exception as e:
            print(f"[ERROR] Interpolation failed at t={t:.2f}: {e}")

    return interpolated_frames


# Process video using FILM interpolation
def process_video(input_path, factor=2):
    print("[INFO] Starting video processing...")
    start_total = time.time()

    model = load_film_model()
    if model is None:
        raise RuntimeError("FILM model could not be loaded.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames.")

    new_frames = []
    num_intermediate = factor - 1

    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        new_frames.append(f1)
        mids = interpolate_frames(model, f1, f2, num_intermediate)
        new_frames.extend(mids)
    new_frames.append(frames[-1])

    height, width, _ = new_frames[0].shape
    out_filename = os.path.splitext(os.path.basename(input_path))[0] + f'_film_{factor}x.mp4'
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    for frame in new_frames:
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

    print(f"[INFO] Processing complete. Saved to {out_path}")
    print(f"[INFO] Total time: {time.time() - start_total:.2f} seconds")
    return out_path


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        vid = request.files.get('video')
        factor = int(request.form.get('factor', 2))
        if vid:
            in_path = os.path.join(app.config['UPLOAD_FOLDER'], vid.filename)
            vid.save(in_path)
            out_path = process_video(in_path, factor)
            return redirect(url_for('download', filename=os.path.basename(out_path)))
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
