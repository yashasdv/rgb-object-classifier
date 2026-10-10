# RGB Color Classifier Using Machine Learning

## 1. Project Overview

Welcome to the RGB Color Classification project!

In this project, you will build a machine learning model that predicts whether an image is predominantly Red, Green, or Blue.

The model will classify images into three classes:

| Label | Color |
|---|---|
| Red | Images that are predominantly red |
| Green | Images that are predominantly green |
| Blue | Images that are predominantly blue |

You will use Python, Google Colab, OpenCV, NumPy, Matplotlib, Scikit-learn, and Joblib.

**Your goal:** Understand the complete machine learning workflow, from loading an image dataset to training, testing, evaluating, and saving a model.

---

## 2. Start Here: Google Colab Notebook

Since this is a custom project, you will create a new notebook from scratch in Google Colab.

### Instructions

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Name your notebook something like `rgb_color_classifier.ipynb`.
3. Follow the code blocks provided in this README and copy them into separate cells in your Colab notebook.
4. Run the cells in order, from top to bottom.
5. Check the output after every important step.
6. Troubleshoot any errors before moving forward.
7. Save your completed notebook.

---

## 3. Learning Objectives

By completing this project, you should be able to:

1. Explain the difference between traditional programming and machine learning.
2. Understand supervised learning and image classification.
3. Organize an image dataset using class labels.
4. Load and visualize images using OpenCV.
5. Resize and normalize images.
6. Convert images into numerical features.
7. Divide a dataset into training and testing subsets.
8. Train a Support Vector Machine (SVM) classifier.
9. Predict labels for images.
10. Evaluate a model using accuracy and a confusion matrix.
11. Save a trained model.

---

## 4. How the Project Works

Your machine learning workflow follows these stages:

```text
Image Dataset (Red, Green, Blue)
     |
     v
Load and Inspect Images
     |
     v
Preprocess Images (Resize, Flatten, Normalize)
     |
     v
Create Features and Labels
     |
     v
Split into Training and Testing Data
     |
     v
Train the SVM Model
     |
     v
Predict Color Labels
     |
     v
Evaluate Accuracy and Confusion Matrix
     |
     v
Test a New Image
     |
     v
Save the Trained Model
```

---

## 5. Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Write and execute Python code in a browser |
| Git | Track changes to project files |
| GitHub | Store and share your project |
| OpenCV | Load, resize, and process images |
| NumPy | Store and manipulate numerical arrays |
| Matplotlib | Display images and evaluation results |
| Scikit-learn | Split data, train the model, and evaluate predictions |
| Joblib | Save the trained model |

---

## 6. Step 1 — Download the Dataset ZIP from GitHub

The RGB color image dataset is provided as a ZIP file in this repository (`dataset.zip`).

### How to do it

1. Locate the `dataset.zip` file in this repository.
2. Download it to your computer.
3. Open your Google Colab notebook.
4. Upload the ZIP file into the Colab session using the steps below.

---

## 7. Step 2 — Install the Required Libraries

Run this in a Colab code cell:

```python
!pip install opencv-python-headless numpy matplotlib scikit-learn joblib
```

Verify the installation in a new cell:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import joblib

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Libraries imported successfully.")
```

---

## 8. Step 3 — Upload and Extract the Dataset ZIP

Run this code in a new cell to upload your dataset:

```python
from google.colab import files

uploaded = files.upload()
```

Select the `dataset.zip` file you downloaded.

Now extract it (in a new cell):

```python
import zipfile, os

with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

# Handle the extra wrapper folder from zipping
root = "dataset"
contents = os.listdir(root)
if len(contents) == 1 and os.path.isdir(os.path.join(root, contents[0])):
    root = os.path.join(root, contents[0])

classes = sorted(os.listdir(root))
print("Using folder:", root)
print("Classes found:", classes)   # should print ['Blue', 'Green', 'Red']
```

---

## 9. Step 4 — Set the Dataset Path

Update the path according to the actual folder structure in Colab.

```python
DATASET_DIR = root if 'root' in globals() else "dataset"

import os
print(os.listdir(DATASET_DIR))
```

---

## 10. Step 5 — Inspect the Dataset

Before training a model, verify that the expected images exist.

```python
CLASSES = ["Red", "Green", "Blue"]

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(folder):
        print(f"Missing folder: {folder}")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"Class {class_name}: {len(images)} images")
```

---

## 11. Step 6 — Understand Images as Numerical Data

View a sample image to see how it's represented.

```python
sample_folder = os.path.join(DATASET_DIR, "Red")

sample_files = [
    f for f in os.listdir(sample_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

sample_path = os.path.join(sample_folder, sample_files[0])
image = cv2.imread(sample_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_image)
plt.title("Sample Red image")
plt.axis("off")
plt.show()

print("Image shape:", image.shape)
```

---

## 12. Step 7 — Load the Complete Dataset and Preprocess

Combine image loading, resizing (to 64x64), flattening, and normalization into one function.

```python
def load_dataset(dataset_dir):
    X = []
    y = []

    # Map classes to integers
    class_map = {"Red": 0, "Green": 1, "Blue": 2}
    
    for class_name in class_map.keys():
        folder = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize to 64x64
            image = cv2.resize(image, (64, 64))
            
            # Flatten to 1D and Normalize (0 to 1)
            features = image.astype(np.float32).flatten() / 255.0

            X.append(features)
            y.append(class_map[class_name])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64)
    )

X, y = load_dataset(DATASET_DIR)

print("Feature matrix shape:", X.shape)
print("Label array shape:", y.shape)
```

---

## 13. Step 8 — Split the Dataset

We will use 80% of the data for training and 20% for testing.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
```

---

## 14. Step 9 — Train the Model

We'll use a Support Vector Machine (SVM) classifier.

```python
from sklearn.svm import SVC

model = SVC(kernel="linear")
model.fit(X_train, y_train)

print("Model training completed.")
```

---

## 15. Step 10 — Calculate Test Accuracy

Evaluate the model on the test dataset.

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")
```

---

## 16. Step 11 — Generate a Confusion Matrix

Visualize the errors and see if the model confuses certain colors.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = [0, 1, 2]
display_labels = ["Red", "Green", "Blue"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
display.plot(cmap="Blues", values_format="d")
plt.title("RGB Classification — Confusion Matrix")
plt.show()
```

---

## 17. Step 12 — Predict a New Image

Upload a custom image to test your newly trained model.

```python
print("Upload a new image to test:")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

test_img = cv2.imread(fname)
test_img_resized = cv2.resize(test_img, (64, 64))
test_input = test_img_resized.flatten() / 255.0

prediction = model.predict([test_input])[0]
class_names = {0: "Red", 1: "Green", 2: "Blue"}

plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title(f"Model predicts: {class_names[prediction]}")
plt.axis('off')
plt.show()
```

---

## 18. Step 13 — Save and Export the Model

Export the model so it can be used for the Live Webcam Application in Week 3!

```python
import joblib

joblib.dump(model, "rgb_model.pkl")
print("Model saved to rgb_model.pkl")

# Download to your local computer
files.download("rgb_model.pkl")
```

---

### 🎉 Congratulations! You have successfully built, evaluated, and saved an RGB Color Classifier! Keep your Colab notebook and the downloaded `rgb_model.pkl` ready for next week.
