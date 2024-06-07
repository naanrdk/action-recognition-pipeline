import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Define the classes
classes = ['Normal', 'Fighting', 'Robbery', 'Shoplifting', 'Stealing']

# Load pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Build the model architecture for action recognition
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load pre-trained weights for action recognition
model.load_weights('action_recognition_weights.h5')

# Function to preprocess input frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Function to predict action class
def predict_action(frame):
    preprocessed_frame = preprocess_frame(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    predictions = model.predict(preprocessed_frame)
    action_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    return action_class, confidence

# Function to detect anomaly behavior
def detect_anomaly(frame):
    action_class, confidence = predict_action(frame)
    
    # Define anomaly thresholds for each class
    anomaly_thresholds = {
        'Normal': 0.2,
        'Fighting': 0.8,
        'Robbery': 0.7,
        'Shoplifting': 0.6,
        'Stealing': 0.6
    }
    
    if confidence < anomaly_thresholds[action_class]:
        return True, action_class
    else:
        return False, action_class

# Function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect anomaly behavior in the current frame
        is_anomaly, action_class = detect_anomaly(frame)
        
        if is_anomaly:
            print(f"Anomaly Detected: {action_class}")
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'video.mp4'
process_video(video_path)
