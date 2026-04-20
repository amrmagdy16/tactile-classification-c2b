# Assignment C2b: Tactile Material Classification from Deformation and Shear Modalities (REAL-SENSOR)

## What to do: Develop AI models for material classification using the mechanical-response modalities of the tactile sensor, namely deformation and shear.
1) Prepare datasets based on deformation and shear data
2) Train classification models for material recognition using these modalities
3) Investigate suitable feature extraction or learning strategies for mechanical-response data
4) Evaluate repeatability and sensitivity to variations in contact conditions
5) Analyze the contribution of deformation and shear information to material discrimination

Software needed: Python, ROS2 (if sensor is integrated through ROS), NumPy, Pandas, scikit-learn, PyTorch or TensorFlow, Jupyter Notebook, signal processing and visualization libraries such as SciPy and Matplotlib

Research needed: Mechanical-response sensing in tactile systems, deformation and shear feature extraction, learning strategies for non-image tactile data, sensitivity analysis under varying contact conditions, material discrimination from force and deformation patterns

Deliverables: Prepared dataset for deformation and shear modalities, trained material classification models, evaluation of repeatability and sensitivity, and analysis of the contribution of mechanical-response signals

# Starting point:
## Interface for Daimon tactile sensor

# How to use

## Work with Python 3.8/3.9/3.10/3.11. Make sure you have cuda toolkit 12.x installed, otherwise you might need to modify setup.py

## Install the package
    pip install .

## Plug in the sensor

## Run
    python main.py
# Baxter:
Fork and use -> https://github.com/giangalv/baxter_rosbridge_adapter, follow the README. 
