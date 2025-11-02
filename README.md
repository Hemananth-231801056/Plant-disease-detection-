ect utilizes a Convolutional Neural Network (CNN), specifically a Transfer Learning approach with the EfficientNetB0 model, to accurately classify and recognize various diseases in plant leaves from images. The goal is to provide a fast and automated method for early disease detection to aid farmers and agricultural experts.

💻 Project Overview
Early and accurate diagnosis of plant diseases is crucial for minimizing crop loss. This project addresses this by building a highly performant image classification model using deep learning. The model is trained to distinguish between 17 different classes of plant leaf conditions, including healthy leaves and various diseases across different plant species.

Key Libraries & Technologies
The core of this project is built using the TensorFlow and Keras deep learning frameworks.

TensorFlow / Keras: For defining, training, and evaluating the CNN model.

EfficientNetB0: Used as the base model for transfer learning, leveraging its powerful feature extraction capabilities.

Numpy: For numerical operations and data handling.

Matplotlib: For data visualization, particularly plotting training history.

split-folders: A utility for easily dividing the dataset into training, validation, and testing sets.

📊 Dataset Description
The model is trained on the "Plant leave diseases dataset with augmentation" dataset. The dataset contains a large collection of augmented images of plant leaves, categorized into multiple classes based on the plant type and the specific disease (or if it's healthy).

Dataset Statistics from Code
Based on the code, the dataset was split using a 80:10:10 ratio:

Train Ratio: 80%

Validation Ratio: 10%

Test Ratio: 10%

The code identified 17 different classes of plant leaf diseases/conditions:

class_names: ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_leaf_spot)', 'Grape___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Target_Spot', 'Tomato___healthy']

🚀 Deep Learning Model
The project employs a Transfer Learning strategy to achieve high accuracy with fewer training epochs.

Model Architecture
Base Model: EfficientNetB0 pre-trained on the ImageNet dataset.

Preprocessing: Images are resized to (160, 160) and preprocessed using tf.keras.applications.efficientnet.preprocess_input.

Top Layers: Added layers for classification:

GlobalAveragePooling2D: Reduces the feature map size.

Dropout(0.2): Regularization to prevent overfitting.

Dense Layer with softmax activation: Outputs probability scores for the 17 classes.

Training Strategy
The model was trained in two phases:

Feature Extraction (Initial Training):

The base_model (EfficientNetB0) layers were frozen (trainable=False).

Only the newly added top classification layers were trained for 3 epochs (initial_epochs = 3).

Fine-Tuning:

The base_model was unfrozen (trainable=True).

The first 100 layers of the base model were re-frozen to preserve the general ImageNet features.

The entire model was re-compiled with a very low learning rate (1e-5) and trained for an additional 5 epochs (fine_tune_epochs = 5).

Final Model Performance
The final model demonstrated strong generalization performance on the held-out test set:

Metric	Result (Code Output)
Test Accuracy	~97.89%

Export to Sheets

🛠️ How to Run the Project
The code is designed to be run in a Google Colab environment.

Upload the Code: Open the provided Python script (untitled2 (1).py) in Google Colab.

Install Dependencies: The script automatically installs required libraries:

Bash

!pip install tensorflow split-folders matplotlib
Execute Cells: Run all the code cells sequentially. The script will handle the following:

Downloading and unzipping the dataset.

Splitting the data into train/validation/test folders.

Creating the data generators.

Building and training the EfficientNetB0 model (Feature Extraction + Fine-Tuning).

Evaluating the model on the test set.

Saving the final model as /content/plant_disease_recognition_model.keras.

➡️ Future Improvements
Deployment: Wrap the model in a web service (e.g., Flask/Streamlit) or a mobile application for real-time inference in the field.

Hyperparameter Optimization: Use tools like Keras Tuner or Optuna for a more rigorous search for the optimal learning rate, dropout rate, and fine-tuning layer cutoff.

Model Comparison: Experiment with other transfer learning models like ResNet or Inception to compare performance against EfficientNetB0.

Quantization: Convert the final model to a TensorFlow Lite model for faster inference on edge devices.
