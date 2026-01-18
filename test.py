import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ปิด Warning ของ TensorFlow ให้หน้าจอสะอาด
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Starting evaluation process...")

# ==========================================
# 1. ตั้งค่า Path และโหลดข้อมูล
# ==========================================
MODEL_PATH = 'glaucoma_model_trained.h5'
DATASET_PATH = 'dataset'

# ถ้าหาโฟลเดอร์ dataset ไม่เจอ ให้ลองหาในโฟลเดอร์ย่อย (เผื่อ Roboflow สร้างชื่อแปลกๆ)
if not os.path.exists(DATASET_PATH):
    found = False
    for d in os.listdir('.'):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'train')):
            DATASET_PATH = d
            found = True
            break
    if not found:
        print("Error: Dataset folder not found.")
        sys.exit(1)

# หาโฟลเดอร์สำหรับ Test (รองรับชื่อ valid, val, test)
test_dir = None
for name in ['test', 'valid', 'val']:
    path = os.path.join(DATASET_PATH, name)
    if os.path.exists(path):
        test_dir = path
        break

if not test_dir:
    # ถ้าไม่มีจริงๆ ให้ใช้ Train แก้ขัดไปก่อนเพื่อทดสอบระบบ
    print("Warning: Test set not found. Using train set for demonstration.")
    test_dir = os.path.join(DATASET_PATH, 'train')

print(f"Testing with data from: {test_dir}")

# ==========================================
# 2. เตรียม Image Generator
# ==========================================
# shuffle=False สำคัญมาก! เพื่อให้ลำดับรูปตรงกับเฉลย
test_datagen = ImageDataGenerator(rescale=1./255)

try:
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
except Exception as e:
    print(f"Error initializing data generator: {e}")
    sys.exit(1)

# ==========================================
# 3. โหลดโมเดลและทำนายผล
# ==========================================
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
    sys.exit(1)

print("Loading model...")
model = load_model(MODEL_PATH)

print("Predicting classes...")
# ทำนายผล (จะได้ค่าความน่าจะเป็น 0.0 - 1.0)
predictions = model.predict(test_generator, verbose=1)

# แปลงค่า: ถ้า > 0.5 ให้เป็น 1, ถ้า <= 0.5 ให้เป็น 0
y_pred = (predictions > 0.5).astype(int).ravel()
y_true = test_generator.classes

# ==========================================
# 4. แสดงผลลัพธ์และกราฟ
# ==========================================
class_labels = list(test_generator.class_indices.keys())

print("\n--- Classification Report ---")
if len(y_true) > 0:
    print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))
else:
    print("Error: No data found to report.")

# วาดกราฟ Confusion Matrix
print("Plotting Confusion Matrix...")
try:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not plot confusion matrix: {e}")
