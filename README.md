# Alzheimer-Brain-Tumour-Breast-Cancer-Covid-19-Heart-Disease-detection-using-machine-learning

This repository contains various machine learning (ML) and deep learning (DL) models designed to detect and predict the following diseases and conditions:

-Alzheimer's Disease
-Brain Tumors
-Breast Cancer
-COVID-19
-Heart Disease

Each disease detection model is based on specific datasets, algorithms, and techniques. The goal of these models is to assist in early diagnosis, which can significantly improve patient outcomes. Below is a brief description of the disease detection models and the technologies used.

* Diseases and Conditions
  
1. Alzheimer's Disease Detection
Objective: Identify early signs of Alzheimer’s Disease (AD) based on brain scans and cognitive tests.
Dataset: Brain imaging datasets (MRI scans).
Techniques: Convolutional Neural Networks (CNN).
Output: Predict the likelihood of Alzheimer’s Disease in patients based on brain images.

2. Brain Tumor Detection
Objective: Detect the presence and type of brain tumors from MRI images.
Dataset: Brain MRI datasets.
Techniques: CNN and using pre-trained models like VGG16.
Output: Predict the tumor type (NO/YES Tomour).

3. Breast Cancer Detection
Objective: Detect breast cancer from mammograms or biopsy data.
Dataset: Breast Cancer datasets.
Techniques: Random Forest.
Output: Classify tumors as malignant or benign.

4. COVID-19 Detection
Objective: Detect COVID-19 infection using chest X-rays, CT scans, or clinical data.
Dataset: COVID-19 Chest X-ray dataset, CT scan images.
Techniques: CNN and Pre trained model(VGG16).
Output: Predict whether a person is infected with COVID-19 based on imaging data.

5. Heart Disease Detection
Objective: Predict the presence of heart disease using patient health data.
Dataset: UCI Heart Disease dataset, Cleveland Heart Disease dataset.
Techniques: Random Forests and XGBOOST technique.
Output: Classify the risk of heart disease (Negative or Positive) based on medical data like Old Peak, Max Heart Rate Achived, Excercise induces angina, No. of major vessels, Type chest Pain, Age and Thal.

* Technologies and Libraries Used
Machine Learning: Scikit-learn, XGBoost
Deep Learning: TensorFlow, Keras, PyTorch
Data Processing: Pandas, NumPy
Image Processing: OpenCV, PIL
Visualization: Matplotlib, Seaborn, Plotly
Model Deployment: Flask

* Features
Pre-trained Models: Use pre-trained models for transfer learning to improve performance on small datasets.
Cross-validation: Implement k-fold cross-validation to evaluate the performance and prevent overfitting.
Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to fine-tune model parameters.
Data Augmentation: For image-based models, use data augmentation techniques to improve the robustness of the model.

* Installation
To run this project please install all the libraries from the requirements.txt file.
pip install -r requirements.txt

* Usage
Data Preprocessing: Prepare your data (images or tabular data) as per the model requirements.
Training: Train the model using the provided scripts or notebooks.
Prediction: Use the trained model to make predictions on unseen data.
Evaluation: Evaluate the model using accuracy, precision, recall, and F1 score for classification tasks or mean squared error for regression tasks.
Model Deployment: Deploy the trained model using Flask for real-time prediction.
