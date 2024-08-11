import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from load_model import load_model
from utilities import fetch_csv_data, render_plot

# Ensure the existence of directories
def verify_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Perform shape prediction using the model
def classify_shape(model, img, device):
    # Preprocess the image to match the model's input requirements
    preprocessing = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = preprocessing(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Predict the shape using the model
    with torch.no_grad():
        prediction = model(img)
        _, label = torch.max(prediction, 1)
    return label.item()

def execute():
    # Load the pre-trained model
    model_file = '/content/drive/MyDrive/curve_project/models/shape_recognition_model.pth'
    model = load_model(model_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create necessary directories for output
    verify_directory('/content/drive/MyDrive/curve_project/output/predictions/')

    # List of CSV files with polylines to predict shapes
    data_files = [
        '/content/drive/MyDrive/curve_project/data/circle.csv',
        '/content/drive/MyDrive/curve_project/data/ellipse.csv',
        '/content/drive/MyDrive/curve_project/data/rectangle.csv',
        '/content/drive/MyDrive/curve_project/data/polygon.csv',
        '/content/drive/MyDrive/curve_project/data/star.csv'
    ]

    # Map model output indices to shape names
    shape_labels = {0: 'Circle', 1: 'Ellipse', 2: 'Rectangle', 3: 'Polygon', 4: 'Star'}

    for data_file in data_files:
        shape_paths = fetch_csv_data(data_file)

        # Process and predict each shape
        for index, polyline in enumerate(shape_paths):
            fig, ax = plt.subplots()

            # Check if polyline has at least one valid shape (two columns for X and Y)
            valid_polyline = True
            for points in polyline:
                if points.shape[1] != 2:
                    print(f"Warning: Invalid polyline with shape {points.shape} in {data_file}. Skipping.")
                    valid_polyline = False
                    break

            if not valid_polyline:
                continue  # Skip invalid polylines

            render_plot([polyline], title=f"Input Shape: {os.path.basename(data_file)}")

            # Convert the plotted shape into an image
            fig.canvas.draw()
            img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = Image.fromarray(img_data)

            # Classify the shape
            predicted_index = classify_shape(model, img, device)
            shape_name = shape_labels[predicted_index]
            plt.title(f"Predicted Shape: {shape_name}")

            # Save the prediction plot
            output_file = f"/content/drive/MyDrive/curve_project/output/predictions/{os.path.basename(data_file).replace('.csv', f'_prediction_{index}.png')}"
            plt.savefig(output_file)
            print(f"Prediction saved at {output_file}")

if __name__ == "__main__":
    execute()
