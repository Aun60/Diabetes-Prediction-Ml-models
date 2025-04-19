# Diabetes Prediction using Machine Learning Models

This project explores the use of various regression and classification models on the **PIMA Diabetes Dataset** to predict the likelihood of diabetes in individuals. Both custom implementations and classical ML techniques are used to assess performance based on standard evaluation metrics.

## 📊 Objective

To implement, evaluate, and compare various regression and classification algorithms using the PIMA Diabetes dataset, including visualizations and hyperparameter tuning.

## 🧠 Models Implemented

- Linear Regression (with one and multiple features)
- Polynomial Regression
- Logistic Regression (from scratch)
- Support Vector Machines (SVM):
  - Hard Margin (Linear Kernel)
  - Soft Margin with Polynomial Kernel
  - Soft Margin with RBF Kernel

## 🗃️ Dataset

- **Source**: PIMA Indians Diabetes Dataset
- **Features**: Glucose, BMI, Age, Insulin, etc.
- **Target**: `Outcome` — binary label (0 = No Diabetes, 1 = Diabetes)

## ⚙️ Preprocessing Steps

- Handled missing values (e.g., zeroes in glucose/insulin) via imputation
- Standardized features for faster convergence
- Converted binary classification into a regression task where required

## 📂 Project Structure

### 1. Linear Regression (One Feature)
- Used Glucose to predict the binary Outcome (treated as regression)
- Implemented gradient descent from scratch
- Visualized cost vs iterations
- Metrics: MSE, R-squared

### 2. Linear Regression (Multiple Features)
- Used all features (multivariate regression)
- Feature scaling, L1 and L2 regularization
- Implemented vectorized gradient descent
- Metrics: MSE, R-squared, Adjusted R-squared

### 3. Polynomial Regression
- Generated polynomial features (degrees 2 and 3)
- Predicted probability of diabetes (regression)
- Visualized polynomial curves
- Metrics: MSE, R-squared

### 4. Logistic Regression (from Scratch)
- Binary classification of diabetes
- Gradient descent implementation
- ROC Curve + AUC visualization
- Metrics: Accuracy, Precision, Recall, F1 Score, AUC

### 5. Hard SVM (Linear Kernel)
- Linear classification using hard-margin SVM
- Visualized support vectors and decision boundary
- Metrics: Accuracy, Confusion Matrix

### 6. Soft SVM with Polynomial Kernel
- Used soft-margin SVM with polynomial kernel (varied degree)
- Visualized decision boundaries
- Metrics: Accuracy, Precision, Recall, F1 Score

### 7. Soft SVM with RBF Kernel
- Soft-margin SVM using RBF kernel
- Tuned gamma and C parameters
- Visualized non-linear decision boundaries
- Metrics: Accuracy, Precision, Recall, F1 Score, AUC

## 🧪 Model Evaluation

- **Regression Metrics**: MSE, R², Adjusted R²
- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score, AUC
- **Tools Used**: Cross-validation, Grid Search for hyperparameter tuning

## 🔍 Comparative Analysis

- Compared all models on performance metrics
- Evaluated regularization effects (L1, L2)
- Analyzed kernel performance (Polynomial vs RBF)
- Discussed pros and cons of each method

## 📈 Visualizations

- Cost function over iterations
- Regression lines and polynomial fits
- ROC curves and AUC
- SVM decision boundaries and support vectors

## 🧰 Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/diabetes-prediction-ml-models.git
cd diabetes-prediction-ml-models

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt

# Run notebooks or scripts
