import os
from deepface import DeepFace
import cv2

# ✅ Optional: make TensorFlow logs quiet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ✅ Emoji mapping for each emotion
emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "fear": "😨",
    "neutral": "😐",
    "disgust": "🤢"
}

print("🎥 Starting AI Emotion Detection... Press 'q' to quit.")

# ✅ Use CAP_DSHOW to fix Windows camera issue
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ✅ Check if camera opened properly
if not cap.isOpened():
    print("❌ Camera not detected! Check your camera access or index.")
    exit()
else:
    print("✅ Camera opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        
        print("❌ Failed to capture frame.")
        break

    try:
        # ✅ Analyze current frame
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # ✅ Handle both return types (list or dict)
        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']
        emoji = emoji_map.get(emotion.lower(), "")
        confidence = result['emotion'][emotion]

        # ✅ Display text + emoji on screen
        label = f"{emotion.capitalize()} {emoji}"
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        # Handle occasional frame analysis errors gracefully
        print(f"⚠️ Frame skipped: {e}")

    # ✅ Show the video feed
    cv2.imshow("AI Emotion Detection 😄", frame)

    # ✅ Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release camera and close window
cap.release()
cv2.destroyAllWindows()
