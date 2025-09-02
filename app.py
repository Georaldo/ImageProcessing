# streamlit_app.py

import os, sys, json, time, uuid
import numpy as np
import pandas as pd
import cv2
import streamlit as st

# Dlib embeddings via face_recognition
import face_recognition

# DeepFace for Facenet512 and ArcFace
from deepface import DeepFace

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("DeepFace:", DeepFace.__version__)
try:
    import face_recognition.api as _fr_api
    print("face_recognition loaded")
except Exception as e:
    print("face_recognition import issue:", e)

# Try to import LBPH (opencv-contrib). If missing, we'll do histogram fallback.
_has_contrib = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
print("OpenCV contrib (LBPH) available:", _has_contrib)


# ==== Config Paths ====
CSV_FILE = "students.csv"
QR_FOLDER = "qrcodes"
EMBED_FOLDER = "embeddings"
META_FOLDER = "metadata"

os.makedirs(QR_FOLDER, exist_ok=True)
os.makedirs(EMBED_FOLDER, exist_ok=True)
os.makedirs(META_FOLDER, exist_ok=True)

# Helper to ensure dtype consistency when reading CSV
def load_students_csv():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, dtype=str)
        # expected columns
        expected = {"student_id","name","course","qr_filename",
                    "dlib_file","facenet_file","arcface_file","opencv_file","meta_file"}
        missing = [c for c in expected if c not in df.columns]
        if missing:
            # upgrade existing CSV by adding new columns
            for c in missing:
                df[c] = ""
        return df
    else:
        cols = ["student_id","name","course","qr_filename",
                "dlib_file","facenet_file","arcface_file","opencv_file","meta_file"]
        return pd.DataFrame(columns=cols)


# ==== Face Utilities ====

def detect_and_crop_face_bgr(frame_bgr, target_size=(160,160)):
    if frame_bgr is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if len(boxes) == 0:
        return None
    top, right, bottom, left = boxes[0]
    face = frame_bgr[top:bottom, left:right]
    if face.size == 0:
        return None
    face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    return face

def l2_normalize(vec, eps=1e-9):
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return (v / n).astype(np.float32)

def aggregate_embeddings(emb_list):
    if not emb_list:
        return None
    M = np.vstack(emb_list).astype(np.float32)
    mean = M.mean(axis=0)
    return l2_normalize(mean)


# ==== Embedding Generators ====

def embed_dlib(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    boxes = [(0, rgb.shape[1], rgb.shape[0], 0)]
    encs = face_recognition.face_encodings(rgb, boxes, num_jitters=1, model="small")
    if len(encs) == 0:
        return None
    return l2_normalize(np.array(encs[0], dtype=np.float32))

def embed_facenet(face_bgr):
    res = DeepFace.represent(face_bgr, model_name="Facenet512", detector_backend="retinaface", enforce_detection=True)
    if not res:
        return None
    emb = np.array(res[0]["embedding"], dtype=np.float32)
    return l2_normalize(emb)

def embed_arcface(face_bgr):
    res = DeepFace.represent(face_bgr, model_name="ArcFace", detector_backend="retinaface", enforce_detection=True)
    if not res:
        return None
    emb = np.array(res[0]["embedding"], dtype=np.float32)
    return l2_normalize(emb)

def embed_opencv(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten().astype(np.float32)
    hist = hist / (np.sum(hist) + 1e-9)
    return l2_normalize(hist)


# ==== Registration (Capture + Multi-backend Embeddings) ====

def register_student_capture(student_id, name, course):
    df = load_students_csv()

    if (df["student_id"] == student_id).any():
        st.error(f"Student ID '{student_id}' already exists.")
        return

    # Create QR code
    import qrcode
    qr_path = os.path.join(QR_FOLDER, f"{student_id}_qr.png")
    qrcode.make(student_id).save(qr_path)
    st.info(f"QR saved -> {qr_path}")

    cap = cv2.VideoCapture(0)
    st.warning("Press the 'Capture Sample' button below several times (5–10) to register, then click 'Finish'.")

    dlib_list, facenet_list, arcface_list, opencv_list = [], [], [], []
    captured = 0

    frame_placeholder = st.empty()
    btn_capture = st.button("Capture Sample")
    btn_finish = st.button("Finish")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        face = detect_and_crop_face_bgr(frame, target_size=(160,160))
        disp = frame.copy()
        if face is not None:
            cv2.rectangle(disp, (10, 60), (10+160, 60+160), (0,255,0), 2)
            disp[60:60+160, 10:10+160] = face

        frame_placeholder.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB")

        if btn_capture:
            if face is None:
                st.warning("No face detected; try again.")
                continue
            try:
                e_dlib = embed_dlib(face)
                if e_dlib is not None: dlib_list.append(e_dlib)
            except Exception as e:
                st.warning(f"Dlib embedding error: {e}")

            try:
                e_fn = embed_facenet(face)
                if e_fn is not None: facenet_list.append(e_fn)
            except Exception as e:
                st.warning(f"Facenet embedding error: {e}")

            try:
                e_af = embed_arcface(face)
                if e_af is not None: arcface_list.append(e_af)
            except Exception as e:
                st.warning(f"ArcFace embedding error: {e}")

            try:
                e_cv = embed_opencv(face)
                if e_cv is not None: opencv_list.append(e_cv)
            except Exception as e:
                st.warning(f"OpenCV embedding error: {e}")

            captured += 1
            st.info(f"Captured sample #{captured} (ok: dlib={len(dlib_list)}, facenet={len(facenet_list)}, arcface={len(arcface_list)}, opencv={len(opencv_list)})")

        if btn_finish:
            break

    cap.release()

    if len(dlib_list)==0 and len(facenet_list)==0 and len(arcface_list)==0 and len(opencv_list)==0:
        st.error("No embeddings captured.")
        return

    agg = {}
    if len(dlib_list):   agg["dlib"]    = aggregate_embeddings(dlib_list)
    if len(facenet_list):agg["facenet"] = aggregate_embeddings(facenet_list)
    if len(arcface_list):agg["arcface"] = aggregate_embeddings(arcface_list)
    if len(opencv_list): agg["opencv"]  = aggregate_embeddings(opencv_list)

    paths = {}
    if len(dlib_list):
        p = os.path.join(EMBED_FOLDER, f"{student_id}_dlib.npy")
        np.save(p, np.vstack(dlib_list).astype(np.float32))
        paths["dlib_file"] = p
    else:
        paths["dlib_file"] = ""

    if len(facenet_list):
        p = os.path.join(EMBED_FOLDER, f"{student_id}_facenet.npy")
        np.save(p, np.vstack(facenet_list).astype(np.float32))
        paths["facenet_file"] = p
    else:
        paths["facenet_file"] = ""

    if len(arcface_list):
        p = os.path.join(EMBED_FOLDER, f"{student_id}_arcface.npy")
        np.save(p, np.vstack(arcface_list).astype(np.float32))
        paths["arcface_file"] = p
    else:
        paths["arcface_file"] = ""

    if len(opencv_list):
        p = os.path.join(EMBED_FOLDER, f"{student_id}_opencv.npy")
        np.save(p, np.vstack(opencv_list).astype(np.float32))
        paths["opencv_file"] = p
    else:
        paths["opencv_file"] = ""

    meta = {
        "student_id": student_id,
        "name": name,
        "course": course,
        "qr_filename": qr_path,
        "templates": {k: v.tolist() for k,v in agg.items()}
    }
    meta_path = os.path.join(META_FOLDER, f"{student_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    row = {
        "student_id": student_id,
        "name": name,
        "course": course,
        "qr_filename": qr_path,
        "dlib_file": paths.get("dlib_file",""),
        "facenet_file": paths.get("facenet_file",""),
        "arcface_file": paths.get("arcface_file",""),
        "opencv_file": paths.get("opencv_file",""),
        "meta_file": meta_path
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    import pickle
    if os.path.exists("registered_embeddings.pkl"):
        with open("registered_embeddings.pkl", "rb") as f:
            registered_db = pickle.load(f)
    else:
        registered_db = {}

    if student_id not in registered_db:
        registered_db[student_id] = {
            "name": name,
            "course": course,
            "embeddings": {}
        }

    for algo, vec in agg.items():
        registered_db[student_id]["embeddings"][algo] = vec

    with open("registered_embeddings.pkl", "wb") as f:
        pickle.dump(registered_db, f)

    st.success(f"Enrollment complete for {name} ({student_id})")
    st.json(row)
    st.info(f"Aggregated templates in: {meta_path}")


# ==== Streamlit UI ====
st.title("🎓 Student Registration with Multi-Backend Face Embeddings")

with st.form("student_form"):
    student_id = st.text_input("Student ID")
    name = st.text_input("Name")
    course = st.text_input("Course")
    submit = st.form_submit_button("Start Registration")

if submit:
    register_student_capture(student_id, name, course)
