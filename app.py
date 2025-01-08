import streamlit as st
from PIL import Image
import numpy as np
import subprocess
import os
from pathlib import Path
import tempfile
from streamlit_image_zoom import image_zoom

def run_yolov7_detection(image_path):
    # Path to your YOLOv7 detect.py script
    detect_script = "yolov7/detect.py"
    # python_path = "env1/Scripts/python.exe"
    # Make sure the script exists
    if not Path(detect_script).exists():
        raise FileNotFoundError(f"{detect_script} not found. Ensure you have the YOLOv7 repository cloned.")
    
    # Create a temporary directory to store the output
    output_dir = "results"
    print(image_path)
    if os.path.exists(image_path):
        print("HELLOOOOOOOOOOOOOOOOOOOOOOOOO")
        # raise FileNotFoundError(f"Temporary image file {image_path} does not exist.")
    if os.path.exists("yolov7/runs/train/yolov710/weights/best.pt"):
        print("BYEEEEEEEEEEEEE")
    # Call the YOLOv7 detection script via subprocess
    command = [
        "python", detect_script, 
        "--weights", "yolov7/runs/train/yolov710/weights/best.pt", 
        "--conf", "0.5", 
        "--img-size", "640", 
        "--source", image_path, 
        "--exist-ok"  # Allow overwriting existing results
    ]

    #     command = [
    #     "python", detect_script, 
    #     "--weights", "yolov7/runs/train/yolov710/weights/best.pt", 
    #     "--conf", "0.5", 
    #     "--img-size", "640", 
    #     "--source", image_path, 
    #     "--project", output_dir,  # Save results in the temporary output directory
    #     "--name", "test_results",
    #     "--exist-ok"  # Allow overwriting existing results
    # ]
    
    # Run the detection script
    result = subprocess.run(command, check=True)
    print(result.stdout)  # To see standard output
    print(result.stderr)  # To see error output

    # Locate the output image in the results/test_results folder
    output_image_dir = os.path.join(output_dir, "test_results")
    if not os.path.exists(output_image_dir):
        raise FileNotFoundError(f"Output directory {output_image_dir} not found.")
        

    # Find the latest .jpg file in the output directory
    output_images = list(Path(output_image_dir).glob("*.jpg"))
    if not output_images:
        raise FileNotFoundError("No output images found in the results directory.")

    # Sort by modification time and return the most recent file
    latest_image = max(output_images, key=os.path.getmtime)
    return str(latest_image)  # Return the full path to the image

st.title("Anomaly Detection for Bangles")
st.markdown("This app uses YOLOv7 model to detect anomalies")

# Sidebar for extra features
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Theme toggle (light/dark mode)
theme_mode = st.sidebar.radio("Theme Mode", ("Light", "Dark"))
st.markdown(f"Current Theme: **{theme_mode}**")

# Option to rename detection labels
label_rename = st.sidebar.text_input("Rename Detection Label", "Bangle")

# Toggle for saving annotated images
save_annotated = st.sidebar.checkbox("Save Annotated Images")

# File uploader
test_image = st.file_uploader("Upload an image of bangles", type=["jpg", "jpeg", "png"])

if test_image is not None:
    st.image(test_image, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting Anomalies ...")
    
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(test_image.read())
        tmp_image_path = tmp_file.name
    
    with st.spinner("Detecting Anomalies ..."):
        # Run YOLOv7 detection
        output_image_path = run_yolov7_detection(tmp_image_path)

    # Display the result
    if os.path.exists(output_image_path):
        result_image = Image.open(output_image_path)
        st.image(result_image, caption="Detected Image with Anomalies", use_container_width=True)

        # Option to save annotated images
        if save_annotated:
            save_path = os.path.join("annotated_results", Path(output_image_path).name)
            os.makedirs("annotated_results", exist_ok=True)
            result_image.save(save_path)
            st.success(f"Annotated image saved to: {save_path}")

        # Display detection metrics
        st.sidebar.markdown("### Detection Metrics")
        st.sidebar.write(f"Confidence Threshold: {confidence_threshold}")
        st.sidebar.write(f"Detection Label: {label_rename}")
    else:
        st.error("An error occurred while detecting anomalies.")

# Footer
st.markdown("---")
st.markdown("Designed by Jovina Wee")
