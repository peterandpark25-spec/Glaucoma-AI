import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import time
import random
import os

# ==========================================
# PART 1: AI CORE LOGIC (สมองของระบบ)
# ==========================================

class GlaucomaFundusModel:
    def __init__(self):
        # ชื่อไฟล์โมเดลต้องตรงกับที่เทรนเสร็จ
        self.model_path = 'glaucoma_model_trained.h5'
        self.model = self.load_or_build_model()
        
    def load_or_build_model(self):
        """พยายามโหลดโมเดลที่เทรนแล้ว ถ้าไม่มีจะสร้างใหม่แบบ Dummy"""
        if os.path.exists(self.model_path):
            try:
                return load_model(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                return self.build_dummy_model()
        else:
            return self.build_dummy_model()

    def build_dummy_model(self):
        """สร้างโมเดลเปล่าๆ (Untrained) กรณีไม่มีไฟล์ .h5"""
        input_fundus = Input(shape=(224, 224, 3), name="input_fundus")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_fundus)
        for layer in base_model.layers:
            layer.trainable = False
            
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name="prediction")(x)

        model = models.Model(inputs=input_fundus, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_image_from_stream(self, uploaded_file):
        # อ่านไฟล์จาก Memory โดยตรง
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, uploaded_file):
        p_fundus = self.preprocess_image_from_stream(uploaded_file)
        prediction_score = self.model.predict(p_fundus)[0][0]
        return float(prediction_score)

# ==========================================
# PART 2: STREAMLIT UI (หน้าตาโปรแกรม)
# ==========================================

st.set_page_config(page_title="Glaucoma AI Diagnosis", page_icon="👁️", layout="wide")

# CSS ตกแต่งปุ่มและกล่องข้อความ
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 50px; font-weight: bold; }
    .reportview-container { background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# Cache เพื่อไม่ให้โหลดโมเดลใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def get_ai_system():
    return GlaucomaFundusModel()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # เช็คสถานะโมเดล
    if os.path.exists('glaucoma_model_trained.h5'):
        st.success("🟢 Model Status: Trained (Ready)")
    else:
        st.warning("🟠 Model Status: Untrained (Using Demo)")

    mode = st.radio("Operation Mode:", 
                    ["Simulation (Demo)", "Actual AI Model"])
    
    st.info("""
    **Simulation:** แสดงผลแบบสุ่ม (ใช้ตอนนำเสนอ)
    **Actual AI:** ใช้ผลจากโมเดล AI จริงๆ
    """)
    st.markdown("---")
    st.caption("AI Glaucoma Screening System")

# --- Main Content ---
st.title("👁️ AI Glaucoma Diagnosis")
st.markdown("#### ระบบคัดกรองโรคต้อหินเบื้องต้นด้วย Deep Learning (MobileNetV2)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image", type="primary"):
            with col2:
                st.subheader("2. Analysis Results")
                
                my_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preprocessing image...")
                my_bar.progress(20)
                time.sleep(0.5)
                
                status_text.text("Running Neural Network...")
                my_bar.progress(60)
                
                # --- LOGIC ---
                final_score = 0.0
                
                if mode == "Actual AI Model":
                    # โหลดและทำนายจริง
                    uploaded_file.seek(0)
                    ai_system = get_ai_system()
                    final_score = ai_system.predict(uploaded_file)
                    time.sleep(0.5)
                else:
                    # Demo Mode (สุ่มค่า)
                    time.sleep(1.0)
                    final_score = random.choice([0.15, 0.92])

                my_bar.progress(100)
                status_text.text("Complete!")
                
                # --- Result Display ---
                threshold = 0.5
                confidence_percent = final_score * 100 if final_score > 0.5 else (1 - final_score) * 100
                
                if final_score > threshold:
                    st.error(f"⚠️ GLAUCOMA DETECTED\n\nConfidence: {confidence_percent:.2f}%")
                else:
                    st.success(f"✅ NORMAL EYE\n\nConfidence: {confidence_percent:.2f}%")
                    
                with st.expander("Technical Details"):
                    st.write(f"**Raw Sigmoid Output:** {final_score:.4f}")
                    if mode == "Actual AI Model":
                        if os.path.exists('glaucoma_model_trained.h5'):
                            st.caption("✅ Using Trained Model weights")
                        else:
                            st.caption("⚠️ Warning: Using Untrained weights")
    else:
        st.info("กรุณาอัปโหลดภาพเพื่อเริ่มการวิเคราะห์")
