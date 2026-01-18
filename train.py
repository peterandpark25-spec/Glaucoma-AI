import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# ==========================================
# 1. การตั้งค่า (Configuration)
# ==========================================
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10  # ปรับเพิ่มได้ถ้าต้องการให้แม่นยำขึ้น (เช่น 20-30)
MODEL_NAME = 'glaucoma_model_trained.h5'

# ปิด Log ของ TensorFlow ที่ไม่จำเป็น
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("กำลังเตรียมข้อมูลและโมเดล...")

# ==========================================
# 2. ดาวน์โหลดข้อมูล (Dataset)
# ==========================================
try:
    from roboflow import Roboflow
    # ดาวน์โหลด Dataset จาก Roboflow
    rf = Roboflow(api_key="GnoGROZlVhtKbAanhV3D")
    project = rf.workspace("valorant-emizs").project("glaucoma-detection-kioac-d5snf")
    version = project.version(1)
    dataset = version.download("folder")
    
    DATASET_PATH = dataset.location
    print(f"Dataset loaded at: {DATASET_PATH}")
    
except ImportError:
    print("Warning: Roboflow library not found. Using local 'dataset' folder.")
    DATASET_PATH = 'dataset'
except Exception as e:
    print(f"Warning: Could not download dataset ({e}). Using local folder.")
    DATASET_PATH = 'dataset'

# ตรวจสอบ path ของ dataset
train_dir = os.path.join(DATASET_PATH, 'train')
val_dir = None

# หา folder validation (รองรับชื่อ valid, val, test)
for name in ['valid', 'val', 'test']:
    if os.path.exists(os.path.join(DATASET_PATH, name)):
        val_dir = os.path.join(DATASET_PATH, name)
        break

if not val_dir:
    val_dir = train_dir # Fallback ถ้าหาไม่เจอ

# ==========================================
# 3. เตรียมข้อมูลภาพ (Data Generator)
# ==========================================
# สร้าง Data Augmentation สำหรับ Train set เพื่อลด Overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation set ไม่ต้อง Augment แค่ปรับ scale
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ==========================================
# 4. สร้างโมเดล (MobileNetV2 Transfer Learning)
# ==========================================
# โหลด MobileNetV2 ที่เทรนมาแล้วจาก ImageNet (ไม่เอาส่วนหัวเดิม)
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False # ล็อกน้ำหนักของ base model ไว้

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3), # ช่วยกัน Overfitting
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Output เป็นค่าความน่าจะเป็น 0-1
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 5. เริ่มเทรนโมเดล (Training)
# ==========================================
print(f"Start training for {EPOCHS} epochs...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    validation_steps=max(1, val_generator.samples // BATCH_SIZE)
)

# บันทึกโมเดล
model.save(MODEL_NAME)
print(f"Model saved as '{MODEL_NAME}'")

# ==========================================
# 6. แสดงกราฟผลลัพธ์ (Plotting)
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))

# กราฟ Accuracy
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# กราฟ Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
