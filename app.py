import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
import pickle
from sklearn import svm
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['ENCODINGS_FOLDER'] = 'face_encodings'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ENCODINGS_FOLDER'], exist_ok=True)

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face recognizer (LBPH algorithm)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def capture_face_images(user_id, num_samples=20):
    """Capture multiple face samples for a user"""
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_folder, exist_ok=True)
    
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Capture - Press SPACE to capture, ESC to exit")
    
    count = 0
    while count < num_samples:
        ret, frame = cam.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces and display count
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Capture - Press SPACE to capture, ESC to exit", frame)
        
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            break
        elif k % 256 == 32:  # SPACE pressed
            if len(faces) == 1:
                # Save the captured face
                face_img = gray[y:y+h, x:x+w]  # Save grayscale image for training
                img_name = f"{user_id}_{count}.jpg"
                cv2.imwrite(os.path.join(user_folder, img_name), face_img)
                count += 1
                print(f"[INFO] {count} images captured for {user_id}")
            else:
                print("[WARNING] No face or multiple faces detected. Try again.")
    
    cam.release()
    cv2.destroyAllWindows()
    return count

def train_face_recognition_model():
    """Train the face recognition model using all user images"""
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    # Loop through each user's folder
    for user_id in os.listdir(app.config['UPLOAD_FOLDER']):
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        
        if not os.path.isdir(user_folder):
            continue
            
        # Assign an ID to this user
        label_ids[user_id] = current_id
        
        # Loop through each image of the user
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            
            # Read image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image to standard size
            image = cv2.resize(image, (200, 200))
            
            faces.append(image)
            labels.append(current_id)
        
        current_id += 1
    
    if len(faces) == 0:
        return False
    
    # Train the face recognizer
    face_recognizer.train(faces, np.array(labels))
    
    # Save the trained model
    model_path = os.path.join(app.config['ENCODINGS_FOLDER'], 'face_recognition_model.yml')
    face_recognizer.save(model_path)
    
    # Save the label mappings
    with open(os.path.join(app.config['ENCODINGS_FOLDER'], 'label_mappings.pkl'), 'wb') as f:
        pickle.dump(label_ids, f)
    
    return True

def recognize_face():
    """Recognize face from webcam"""
    # Load the trained model if it exists
    model_path = os.path.join(app.config['ENCODINGS_FOLDER'], 'face_recognition_model.yml')
    
    if not os.path.exists(model_path):
        return None, 0
    
    face_recognizer.read(model_path)
    
    # Load label mappings
    with open(os.path.join(app.config['ENCODINGS_FOLDER'], 'label_mappings.pkl'), 'rb') as f:
        label_ids = pickle.load(f)
    
    # Create reverse mapping (id -> name)
    id_labels = {v: k for k, v in label_ids.items()}
    
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Recognition - Press SPACE to authenticate, ESC to exit")
    
    recognized_user = None
    confidence = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces and predict
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Predict using the trained model
            label, conf = face_recognizer.predict(face_roi)
            confidence = 100 - conf  # Convert to confidence percentage
            
            if label in id_labels:
                recognized_name = id_labels[label]
                label_text = f"{recognized_name} ({confidence:.2f}%)"
                
                # Display name and confidence
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                label_text = "Unknown"
                cv2.putText(frame, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Face Recognition - Press SPACE to authenticate, ESC to exit", frame)
        
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            break
        elif k % 256 == 32:  # SPACE pressed
            if len(faces) == 1 and label in id_labels:
                recognized_user = id_labels[label]
                break
            else:
                print("[WARNING] No face or multiple faces detected. Try again.")
    
    cam.release()
    cv2.destroyAllWindows()
    
    return recognized_user, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        if not user_id:
            flash('User ID is required!', 'error')
            return redirect(request.url)
        
        # Check if user already exists
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        if os.path.exists(user_folder):
            flash('User ID already exists!', 'error')
            return redirect(request.url)
        
        # Capture face images
        num_captured = capture_face_images(user_id)
        
        if num_captured > 0:
            # Train the model with new data
            if train_face_recognition_model():
                flash(f'Registration successful! {num_captured} face images captured.', 'success')
            else:
                flash('Registration completed but model training failed. Please try again.', 'warning')
            return redirect(url_for('index'))
        else:
            flash('Failed to capture face images. Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Recognize face
        user_id, confidence = recognize_face()
        
        if user_id and confidence > 70:  # Confidence threshold (70%)
            session['user_id'] = user_id
            session['confidence'] = confidence
            flash(f'Authentication successful! Welcome {user_id}. Confidence: {confidence:.2f}%', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Authentication failed! Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', 
                          user_id=session['user_id'], 
                          confidence=session['confidence'])

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('confidence', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)