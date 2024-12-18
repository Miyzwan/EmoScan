import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import gc
import threading

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="EmoScan: Real-Time Emotion Detector", 
    page_icon="😊", 
    #layout="wide"
)

# Styling Khusus
st.markdown("""
<style>
.title {
    text-align: center;
    color: #3498db;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 20px;
}
.subtitle {
    text-align: center;
    color: #2ecc71;
    font-size: 1.2em;
}
.emotion-display {
    background-color: rgba(52, 152, 219, 0.1);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.emotion-label {
    font-size: 2em;
    font-weight: bold;
    color: #3498db;
}
.confidence-label {
    font-size: 1.2em;
    color: #2ecc71;
}
</style>
""", unsafe_allow_html=True)

# Nonaktifkan warning TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- STEP 1: Load Model dan Face Cascade ---
@st.cache_resource
def load_resources():
    model = load_model('model_ekspresi_wajah_74.h5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_resources()
class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- STEP 2: Preprocessing Wajah ---
def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.equalizeHist(face_img)
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=(0, -1))
    return face_img

# --- STEP 3: Prediksi Emosi (Dioptimasi) ---
def predict_emotion(face_img, model, confidence_threshold=0.5):
    try:
        processed_img = preprocess_face(face_img)
        predictions = model.predict(processed_img, verbose=0)
        confidence = float(np.max(predictions))
        emotion = class_labels[np.argmax(predictions)] if confidence > confidence_threshold else "Uncertain"
        return emotion, confidence
    except Exception as e:
        st.error(f"Error in emotion prediction: {e}")
        return "Error", 0.0

# --- Fungsi Sidebar Interaktif ---
def create_sidebar():
    with st.sidebar:
        st.image("logo.jpg", width=300)
        st.markdown("<h1 class='title'>EmoScan</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Real-Time Emotion Analysis</p>", unsafe_allow_html=True)
        
        # Panduan Penggunaan
        with st.expander("🚀 Cara Penggunaan"):
            st.write("""
            1. Pastikan webcam terhubung
            2. Tekan tombol 'Mulai Deteksi'
            3. Hadapkan wajah ke kamera
            4. Amati perubahan ekspresi
            """)
        
        # Informasi Emosi
        with st.expander("😀 Tentang Deteksi Emosi"):
            st.write("""
            Aplikasi mendeteksi 6 kategori emosi:
            - Senang (Happy)
            - Marah (Angry)
            - Sedih (Sad)
            - Netral (Neutral)
            - Terkejut (Surprise)
            - Ahegao
            """)
        
        # Kontrol Sensitifitas
        st.subheader("Pengaturan Deteksi")
        confidence_threshold = st.slider(
            "Tingkat Kepercayaan Deteksi", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.5, 
            step=0.1
        )
        
        return confidence_threshold

# --- Variabel Global untuk Threading ---
stop_event = threading.Event()
latest_frame = None
latest_emotion = "Waiting"
latest_confidence = 0.0

# --- Fungsi Pembacaan Frame ---
def read_frames(cap):
    global latest_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Set resolusi dan optimasi frame
            frame = cv2.resize(frame, (640, 480))
            latest_frame = frame
        else:
            break

# --- Fungsi Proses Emosi ---
def process_emotions(model, face_cascade, confidence_threshold):
    global latest_frame, latest_emotion, latest_confidence
    frame_count = 0
    
    while not stop_event.is_set():
        if latest_frame is not None:
            frame_count += 1
            
            # Lakukan prediksi setiap 5 frame untuk mengurangi beban
            if frame_count % 5 == 0:
                # Deteksi wajah
                gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
                
                # Proses setiap wajah
                for (x, y, w, h) in faces:
                    face_roi = latest_frame[y:y+h, x:x+w]
                    emotion, confidence = predict_emotion(face_roi, model, confidence_threshold)
                    
                    # Update global emotion
                    latest_emotion = emotion
                    latest_confidence = confidence
            
            time.sleep(0.01)  # Tambahkan sedikit delay untuk mengurangi beban CPU

# --- Fungsi Utama Streamlit ---
def main():
    # Sidebar Interaktif
    confidence_threshold = create_sidebar()
    
    # Kolom Utama
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<h2 class='title'>Deteksi Emosi Real-Time</h2>", unsafe_allow_html=True)
        frame_window = st.empty()
        fps_placeholder = st.empty()
    
    with col2:
        st.markdown("<div class='emotion-display'>", unsafe_allow_html=True)
        emotion_label = st.empty()
        confidence_label = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tombol Kontrol
    start_col, stop_col = st.columns(2)
    with start_col:
        run_webcam = st.button("🟢 Mulai Deteksi", use_container_width=True)
    with stop_col:
        stop_webcam = st.button("🔴 Hentikan Deteksi", use_container_width=True)
    
    # Proses Deteksi
    if run_webcam:
        # Buka kamera dengan pengaturan optimal
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Reset stop event
        stop_event.clear()

        # Jalankan thread pembacaan frame
        frame_thread = threading.Thread(target=read_frames, args=(cap,))
        frame_thread.start()

        # Jalankan thread proses emosi
        emotion_thread = threading.Thread(target=process_emotions, args=(model, face_cascade, confidence_threshold))
        emotion_thread.start()

        # Hitung FPS
        start_time = time.time()
        frame_count = 0

        try:
            while not stop_event.is_set() and not stop_webcam:
                if latest_frame is not None:
                    frame = latest_frame.copy()
                    
                    # Tambahkan label emosi ke frame
                    color = (0, 255, 0) if latest_confidence > 0.6 else (0, 255, 255) if latest_confidence > 0.4 else (0, 0, 255)
                    label = f"{latest_emotion} ({latest_confidence:.2f})"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Tampilkan frame
                    frame_window.image(frame, channels="BGR")

                    # Update label emosi
                    emotion_label.markdown(f"<p class='emotion-label'>{latest_emotion}</p>", unsafe_allow_html=True)
                    confidence_label.markdown(f"<p class='confidence-label'>Confidence: {latest_confidence:.2f}</p>", unsafe_allow_html=True)

                    # Hitung FPS
                    frame_count += 1
                    if time.time() - start_time > 1:
                        fps = frame_count / (time.time() - start_time)
                        fps_placeholder.metric("Frame Rate", f"{fps:.2f} FPS")
                        start_time = time.time()
                        frame_count = 0

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # Hentikan thread dan bersihkan sumber daya
            stop_event.set()
            cap.release()
            gc.collect()
            st.write("Webcam stopped.")

# Jalankan Aplikasi
if __name__ == "__main__":
    main()