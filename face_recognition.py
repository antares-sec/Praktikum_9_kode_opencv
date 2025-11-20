import os

# Konfigurasi tensorflow agar berjalan menggunakan CPU
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')

import cv2 as cv
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np
import time


def load_model(model_path):
    """Load MTCNN detector and Keras recognition model.

    Returns (detector, recognition_model) or (None, None) on failure.
    """
    try:
        detector = MTCNN()
        recognition = tf.keras.models.load_model(model_path)
        return detector, recognition
    except Exception as e:
        print("Error loading models:", e)
        return None, None


def preprocess_face(face_rgb, target_size):
    """Preprocess a face crop for the recognition model.

    - face_rgb: HxWx3 uint8 or float image in RGB space.
    - target_size: (height, width)
    Returns a float32 array shaped (1, h, w, 3) scaled to [0,1].
    """
    face = face_rgb.astype('float32')
    if face.max() > 2.0:
        face = face / 255.0
    face_resized = tf.image.resize(face, target_size).numpy()
    return np.expand_dims(face_resized.astype('float32'), axis=0)


def run_webcam(model_path='face_cnn.keras', camera_index=0):
    detector, recognition = load_model(model_path)
    if detector is None or recognition is None:
        print('Failed to load detector or recognition model. Exiting.')
        return

    try:
        input_shape = recognition.input_shape
        target_h, target_w = int(input_shape[1]), int(input_shape[2])
    except Exception:
        target_h, target_w = 160, 160

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Cannot open camera', camera_index)
        return

    print('Press q to quit')
    fps_time = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print('Failed to grab frame')
                break

            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            img_for_detector = (frame_rgb).astype('uint8')

            detections = detector.detect_faces(img_for_detector)

            for det in detections:
                x, y, w, h = det.get('box', (0, 0, 0, 0))
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = max(0, int(x + w))
                y2 = max(0, int(y + h))

                face_rgb = frame_rgb[y1:y2, x1:x2]
                if face_rgb.size == 0:
                    continue

                face_input = preprocess_face(face_rgb, (target_h, target_w))

                # Run model
                try:
                    preds = recognition.predict(face_input)
                    if preds.ndim == 2 and preds.shape[1] > 1:
                        class_id = int(np.argmax(preds[0]))
                        conf = float(np.max(preds[0]))
                        label_text = f'{class_id} ({conf*100:.1f}%)'
                    else:
                        label_text = 'pred'
                except Exception as e:
                    label_text = 'err'
                    print('Prediction error:', e)

                # Draw rectangle and label on BGR frame
                cv.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame_bgr, label_text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show FPS
            now = time.time()
            fps = 1.0 / (now - fps_time) if now != fps_time else 0.0
            fps_time = now
            cv.putText(frame_bgr, f'FPS: {fps:.1f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv.imshow('Webcam - Face Recognition', frame_bgr)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    model_path = 'face_cnn.keras'
    run_webcam(model_path=model_path, camera_index=0)
