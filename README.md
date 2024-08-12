# Curvetopia

Welcome to **Curvetopia** – a cutting-edge toolkit for 2D shape analysis. This project offers a robust suite of tools designed to streamline the process of shape analysis, from detecting and classifying geometric shapes to advanced tasks like symmetry detection and shape recognition using deep learning.

## Key Features

- **Shape Regularization:** Automatically classify shapes such as circles, ellipses, rectangles, polygons, and stars.
- **Symmetry Detection:** Identify reflective and rotational symmetries within 2D shapes.
- **Curve Completion:** Use advanced interpolation techniques to complete incomplete shapes.
- **Shape Recognition:** Train and utilize a deep learning model to recognize shapes from images of their polylines.

## Project Structure
```plaintext
curvetopia/
│
├── data/
│   ├── circle.csv
│   ├── ellipse.csv
│   ├── rectangle.csv
│   ├── polygon.csv
│   ├── star.csv
│   ├── frag0.csv
│   ├── frag1.csv
│   ├── occlusion1.csv
│   ├── occlusion2.csv
│
├── models/
│   └── shape_recognition_model.pth
│
├── scripts/
│   ├── 1.py
│   ├── regularization.py
│   ├── symmetry.py
│   ├── fullcurve.py
│   ├── trainmodel.py
│   ├── utilities.py
│   ├── shapeg.py
│   └── employmodel.py
│
└── output/
    └── predictions/

```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/curvetopia.git
cd curvetopia
```

### 2. Install Dependencies

Install the required libraries using pip:

```bash
pip install numpy matplotlib scipy torch torchvision pillow scikit-learn
```
### 3. Mount Google Drive (Optional, for Colab users)

If you're using Google Colab, you can mount your Google Drive to save outputs and access datasets:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Prepare Your Dataset

Ensure your dataset is in CSV format, with each row representing a point in a shape (X, Y coordinates). Place your datasets in the `data/` directory.


### 5. Execute the Scripts

Run the scripts in the following sequence to process the data, train the model, and make predictions:

- **Step 1:** Regularize Shapes  
  Use `regularization.py` to identify and classify geometric shapes in your dataset.
  ```bash
  python scripts/regularization.py
  ```
- **Step 2:** Detect Symmetry  
  Use `symmetry.py` to analyze the shapes for reflective and rotational symmetries.
  ```bash
  python scripts/symmetry.py
  ```
- **Step 3:** Complete Curves  
  Use `fullcurve.py` to interpolate and complete any missing parts of the shapes.
  ```bash
  python scripts/fullcurve.py
  ```
- **Step 4:** Train the Shape Recognition Model

   Use `trainmodel.py` to train a ResNet-18 model on your shapes.

   ```bash
   python scripts/trainmodel.py
   ```
- **Step 5:** Load and Deploy the Model
 
   Use `employmodel.py` to load the trained model and make predictions on new shape data:

   ```bash
   python scripts/employmodel.py
   ```
### 6. View Predictions

Predictions will be saved in the `output/predictions/` directory. You can visualize the results using any image viewer or within your Python environment.

### Conclusion

**Curvetopia** provides a comprehensive toolkit for 2D shape analysis, encompassing geometric classification, symmetry detection, curve completion, and deep learning-based shape recognition. This end-to-end pipeline is designed for flexibility and can be adapted for various research, educational, or practical applications in computer vision and shape analysis.



