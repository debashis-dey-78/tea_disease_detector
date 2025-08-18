# Library imports
import numpy as np
import streamlit as st
import cv2
import os
import gdown # Ensure gdown is in your requirements.txt for Hugging Face
# from tensorflow.keras.models import load_model

# Import the TFLite interpreter
try:
    # Recommended for deployment: smaller package
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback for local testing: full TensorFlow package
    import tensorflow.lite as tflite
    

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Two-Stage Leaf Analysis")

# --- Model Paths and IDs ---
MODEL_PATH_LNL = "stage1_nonleaf.tflite"
FILE_ID_LNL = "1HNaGUmJyxu5JdYCn_p4kVj9MNYKJn_Sk"
URL_LNL = f"https://drive.google.com/uc?id={FILE_ID_LNL}"

MODEL_PATH_DISEASE = "stage2_Tea_disease.tflite"
FILE_ID_DISEASE = "12dFv5o5tZYkn0pzNeWHKnjXyWutD7hPZ"
URL_DISEASE = f"https://drive.google.com/uc?id={FILE_ID_DISEASE}"

# Function to download and load models (MODIFIED FOR TFLITE)
@st.cache_resource # Cache the loaded models
def load_all_models():
    interpreters_loaded = {}
    
    # --- Download and Load Model 1: Leaf vs Non-Leaf ---
    if not os.path.exists(MODEL_PATH_LNL):
        with st.spinner(f"Downloading Leaf/Non-Leaf model ({MODEL_PATH_LNL})..."):
            try:
                gdown.download(URL_LNL, MODEL_PATH_LNL, quiet=False)
            except Exception as e:
                st.error(f"Error downloading {MODEL_PATH_LNL}: {e}")
    
    # Load the TFLite model into an interpreter
    try:
        interpreter_lnl = tflite.Interpreter(model_path=MODEL_PATH_LNL)
        interpreter_lnl.allocate_tensors() # IMPORTANT: Allocate memory
        interpreters_loaded['lnl'] = interpreter_lnl
    except Exception as e:
        st.error(f"Error loading model {MODEL_PATH_LNL}: {e}")
        interpreters_loaded['lnl'] = None
    
    # --- Download and Load Model 2: Tea Disease ---
    if not os.path.exists(MODEL_PATH_DISEASE):
        with st.spinner(f"Downloading Tea Disease model ({MODEL_PATH_DISEASE})..."):
            try:
                gdown.download(URL_DISEASE, MODEL_PATH_DISEASE, quiet=False)
            except Exception as e:
                st.error(f"Error downloading {MODEL_PATH_DISEASE}: {e}")

    # Load the TFLite model into an interpreter
    try:
        interpreter_disease = tflite.Interpreter(model_path=MODEL_PATH_DISEASE)
        interpreter_disease.allocate_tensors() # IMPORTANT: Allocate memory
        interpreters_loaded['disease'] = interpreter_disease
    except Exception as e:
        st.error(f"Error loading model {MODEL_PATH_DISEASE}: {e}")
        interpreters_loaded['disease'] = None

    return interpreters_loaded.get('lnl'), interpreters_loaded.get('disease')

# This line now loads interpreter objects, not Keras models
interpreter_leaf_non_leaf, interpreter_disease = load_all_models()

############################################################################################# UI SECTION ############################################################

# Class names for Tea Disease Model
CLASS_NAMES_DISEASE = ['bb', 'gl', 'rr', 'rsm']

# Disease Information for Tea Disease Model
disease_info = {
    'gl': """
        **This is a Non-diseased (Healthy) tea leaf.**
        No specific management actions are required other than routine good agricultural practices to maintain plant health.
    """,
    'rr': """
        **Disease: Red Rust** **Description:** Red rust is a common disease of tea plants caused by an alga, *Cephaleuros virescens*. It appears as orange-brown, velvety patches on leaves and can sometimes affect stems, leading to dieback if severe.
        **Management:** * Improve air circulation by proper pruning and spacing.
        * Manage shade to reduce humidity.
        * Apply appropriate copper-based fungicides if the infestation is severe and widespread, following local agricultural guidelines.
        * Ensure balanced plant nutrition.
    """,
    'rsm': """
        **Pest: Red Spider Mites** **Description:** Red spider mites (*Oligonychus coffeae*) are common pests that suck sap from tea leaves, leading to a reddish-brown or bronze discoloration and sometimes fine webbing, especially on the undersides of leaves. Severe infestations can reduce yield and quality.
        **Management:** * Maintain plant vigor through proper irrigation and fertilization.
        * Encourage natural predators of mites.
        * Use approved miticides if infestation is heavy and causing economic damage. Consider spot treatments.
        * Regularly monitor for early signs of infestation.
    """,
    'bb': """
        **Disease: Brown Blight** **Description:** Brown blight, caused by the fungus *Colletotrichum spp.* (often *Colletotrichum camelliae*), leads to distinct brown, often circular or irregular lesions on tea leaves. These lesions may have concentric rings and can enlarge, causing defoliation.
        **Management:** * Prune and destroy affected plant parts to reduce inoculum.
        * Ensure proper plant spacing for good air circulation.
        * Apply fungicides (e.g., mancozeb, copper-based) as recommended by local agricultural experts, especially during wet conditions favorable for fungal growth.
        * Avoid excessive nitrogen fertilization.
    """
}

# --- UI Setup ---
st.title("🌿 Two-Stage Leaf Analysis")
st.markdown("Upload an image of a tea leaf. The system will first check if it's a valid leaf image and then identify potential diseases or pests.")
st.markdown("---")

# --- Upload Options on Main Page ---
st.markdown("### 1. Provide a Leaf Image")
col1, col2 = st.columns(2)
with col1:
    plant_image = st.file_uploader("Upload from device", type=["png", "jpg", "jpeg", "webp","heic"], help="Browse and select an image file (PNG, JPG, JPEG, WEBP,heic).")
with col2:
    captured_image = st.camera_input("Capture with camera", help="Use your device's camera to take a photo of the leaf.")

# Placeholder for all dynamic results and processing messages
results_placeholder = st.empty()

def show_disease_info_main(result_class):
    with st.expander(f"Learn more about: {result_class.upper()}", expanded=True):
        if result_class in disease_info:
            st.markdown(disease_info[result_class])
        else:
            st.warning("No detailed information available for this classification.")


#####################################################################################################

# Main processing logic (MODIFIED FOR TFLITE)
if interpreter_leaf_non_leaf is not None and interpreter_disease is not None:
    image_data = None
    source_info = ""

    if plant_image is not None:
        image_data = plant_image.read()
        source_info = f"Source: Uploaded file (`{plant_image.name}`)"
    elif captured_image is not None:
        image_data = captured_image.read()
        source_info = "Source: Image captured via camera"

    if image_data:
        with results_placeholder.container():
            st.markdown("### 2. Analysis Results")
            st.write(source_info)

            file_bytes = np.asarray(bytearray(image_data), dtype=np.uint8)
            opencv_image_bgr = cv2.imdecode(file_bytes, 1)

            st.image(opencv_image_bgr, channels="BGR", caption="Original Input Image", width=300)
            
            st.write("---")
            st.info("⚙️ Stage 1: Verifying if the image is a leaf...")

            # --- Stage 1 Prediction ---
            opencv_image_rgb = cv2.cvtColor(opencv_image_bgr, cv2.COLOR_BGR2RGB)
            resized_lnl = cv2.resize(opencv_image_rgb, (160, 160))
            normalized_lnl = resized_lnl / 255.0
            input_lnl = np.expand_dims(normalized_lnl, axis=0).astype(np.float32)

            # Get input and output details
            input_details_lnl = interpreter_leaf_non_leaf.get_input_details()
            output_details_lnl = interpreter_leaf_non_leaf.get_output_details()

            # Set the tensor, invoke, and get output
            interpreter_leaf_non_leaf.set_tensor(input_details_lnl[0]['index'], input_lnl)
            interpreter_leaf_non_leaf.invoke()
            prediction_lnl = interpreter_leaf_non_leaf.get_tensor(output_details_lnl[0]['index'])
            
            prediction_lnl_value = prediction_lnl[0][0] 
            is_leaf = prediction_lnl_value <= 0.5 
            leaf_confidence = (1 - prediction_lnl_value if is_leaf else prediction_lnl_value) * 100

            if is_leaf:
                st.success(f"✅ Stage 1 Result: Image identified as a LEAF (Confidence: {leaf_confidence:.2f}%)")
                st.info("⚙️ Stage 2: Checking for tea diseases/pests...")
                st.write("---")

                # --- Stage 2 Prediction ---
                resized_disease = cv2.resize(opencv_image_bgr, (512, 512))
                normalized_disease = resized_disease / 255.0
                input_disease = np.expand_dims(normalized_disease, axis=0).astype(np.float32)

                # Get input and output details
                input_details_disease = interpreter_disease.get_input_details()
                output_details_disease = interpreter_disease.get_output_details()

                # Set the tensor, invoke, and get output
                interpreter_disease.set_tensor(input_details_disease[0]['index'], input_disease)
                interpreter_disease.invoke()
                prediction_disease = interpreter_disease.get_tensor(output_details_disease[0]['index'])

                result_class_disease = CLASS_NAMES_DISEASE[np.argmax(prediction_disease)]
                confidence_disease = np.max(prediction_disease) * 100

                # ... (The rest of your UI code for displaying results remains the same) ...
                st.subheader(f"🔍 Stage 2 Result: Disease/Pest Detection")
                if result_class_disease == 'gl':
                    st.markdown(f'#### <span style="color:green;">Status: Healthy Leaf</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'#### <span style="color:red;">Status: {result_class_disease.upper()}</span> (Potential Issue)', unsafe_allow_html=True)
                
                progress_bar_color = "green" if result_class_disease == 'gl' else "red"
                st.markdown(f"""
                <div style="margin-bottom: 5px;">Confidence: {confidence_disease:.2f}%</div>
                <div style="background-color: #e0e0e0; border-radius: 5px; padding: 3px;">
                    <div style="background-color: {progress_bar_color}; width: {confidence_disease}%; height: 20px; border-radius: 3px; text-align: center; color: white; font-weight: bold;">
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                show_disease_info_main(result_class_disease)

            else:
                st.error(f"❌ Stage 1 Result: Image does NOT appear to be a leaf (Confidence of being Non-Leaf: {leaf_confidence:.2f}%).")
                st.warning("Please upload a clear image of a single tea leaf for accurate analysis.")
                st.markdown("If you believe this is an error, the image might be of poor quality, contain multiple objects, or not be focused on a leaf.")

elif not interpreter_leaf_non_leaf or not interpreter_disease:
    st.error("⚠️ One or both AI models could not be loaded. The application cannot proceed.")
    st.warning("This might be due to issues reaching the model files or an internal error. Please try refreshing the page. If the problem persists, the application maintainer should check the logs.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-size: smaller;'>Developed by Harjinder Singh</p>", unsafe_allow_html=True)