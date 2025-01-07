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
    python_path = "env1/Scripts/python.exe"
    # Make sure the script exists
    if not Path(detect_script).exists():
        raise FileNotFoundError(f"{detect_script} not found. Ensure you have the YOLOv7 repository cloned.")
    
    # Create a temporary directory to store the output
    output_dir = "results"

    # Call the YOLOv7 detection script via subprocess
    command = [
        python_path, detect_script, 
        "--weights", "yolov7/runs/train/yolov710/weights/best.pt", 
        "--conf", "0.5", 
        "--img-size", "640", 
        "--source", image_path, 
        "--project", output_dir,  # Save results in the temporary output directory
        "--name", "test_results",
        "--exist-ok"  # Allow overwriting existing results
    ]
    
    # Run the detection script
    subprocess.run(command, check=True)

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
    
    # # The results will be saved in the output_dir/test_results folder
    # output_image_path = os.path.join(output_dir, "test_results")  # Assuming single image output
    # return output_image_path

st.title("Anomaly Detection for Bangles")
st.markdown("This app uses YOLOv7 model to detect anomalies")

test_image = st.file_uploader("Upload an image of bangles", type=["jpg", "jpeg", "png"])

if test_image is not None:
    st.image(test_image, caption="Uploaded Image")
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
        st.image(result_image, caption="Detected Image with Anomalies")
    else:
        st.error("An error occurred while detecting anomalies.")
