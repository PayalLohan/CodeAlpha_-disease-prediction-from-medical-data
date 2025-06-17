

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # XGBoost is often part of gradient boosting
from xgboost import XGBClassifier # Explicitly importing XGBoost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# --- 1. Load the Dataset ---
# IMPORTANT: Replace 'your_dataset.csv' with the actual path to your downloaded dataset.
# For example, if you download the UCI Heart Disease dataset, it might be a .csv or .data file.
# You might need to add 'names=' parameter if your CSV doesn't have a header.
try:
    # Example for a common dataset like Heart Disease from UCI (often without header)
    # You'll need to find the exact column names or infer them for your chosen dataset.
    # For demonstration, let's assume a simplified dataset structure.
    # If your dataset has a header, just use: df = pd.read_csv('your_dataset.csv')
    
    # Placeholder for a generic dataset. You'll need to adapt this.
    # Let's assume a dataset where the last column is the target variable.
    # Example for UCI Heart Disease (Cleveland dataset) - you might need to combine files or specify separator
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)
    # columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'acc', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    # df.columns = columns
    # df = df.replace('?', np.nan) # Handle missing values represented as '?'
    # df = df.dropna()
    # df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0) # Binarize target for binary classification
    
    # A more general example assuming a CSV with a header and 'target' column
    # For demonstration, let's create a dummy dataset if you don't have one handy
    print("Please make sure to replace 'your_dataset.csv' with the actual path to your dataset.")
    print("If you don't have a dataset, this script will create a small dummy one for demonstration.")

    # Dummy dataset creation (for demonstration purposes if you don't have a real one)
    data = {
        'age': np.random.randint(20, 80, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'symptom_A': np.random.randint(0, 2, 100),
        'blood_pressure': np.random.randint(90, 180, 100),
        'cholesterol': np.random.randint(150, 300, 100),
        'blood_sugar': np.random.randint(70, 200, 100),
        'target_disease': np.random.randint(0, 2, 100) # 0 for no disease, 1 for disease
    }
    df = pd.DataFrame(data)
    print("\n--- Using a Dummy Dataset for Demonstration ---")
    print(df.head())
    
    # If you have a real dataset, uncomment and modify this line:
    # df = pd.read_csv('path/to/your/actual_medical_dataset.csv') 

except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure 'your_dataset.csv' is in the correct path or provide the correct path.")
    exit() # Exit if the dataset isn't found and we're not using a dummy

# --- 2. Preprocessing ---

# Separate features (X) and target (y)
# IMPORTANT: Adjust 'target_disease' to the actual name of your target column
TARGET_COLUMN = 'target_disease' # Adjust this to your dataset's target column name

if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Handle categorical features (e.g., 'gender' in dummy dataset)
# For real datasets, you might have more categorical features.
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    print(f"Encoded categorical column: {col}")

# Handle missing values (if any). For simplicity, we'll use mean imputation for numerical.
# You might need more sophisticated methods based on your data (e.g., KNN imputer, mode for categorical).
if X.isnull().sum().any():
    print("\nHandling missing values...")
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].mean())
                print(f"Imputed missing values in numerical column '{col}' with mean.")
            else:
                # For categorical, use mode if not handled by LabelEncoder before
                X[col] = X[col].fillna(X[col].mode()[0])
                print(f"Imputed missing values in categorical column '{col}' with mode.")

# Scale numerical features
# It's crucial for distance-based algorithms like SVM and Logistic Regression
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("\nScaled numerical features.")

# --- 3. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- 4. Train and Evaluate Different Models ---

models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Support Vector Machine (SVM)": SVC(random_state=42, probability=True), # probability=True for ROC AUC
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # Suppress warning for eval_metric
}

results = {}

print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0) # zero_division=0 to handle cases where no positive predictions are made
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC requires probability scores
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = "N/A" # Some models (like SVC without probability=True) don't have predict_proba by default

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }
    
    print(f"{name} Results:")
    for metric, value in results[name].items():
        print(f"  {metric}: {value:.4f}")

# Display all results in a summary table
print("\n--- Summary of Model Performance ---")
results_df = pd.DataFrame.from_dict(results, orient='index')
print(results_df.round(4))

# --- 5. Basic Prediction Example ---
print("\n--- Demonstration of a Single Prediction ---")
# Let's take the first sample from the test set for a prediction example
sample_index = 0
sample_features = X_test.iloc[sample_index].values.reshape(1, -1)
true_label = y_test.iloc[sample_index]

print(f"Features of sample {sample_index}:\n{X_test.iloc[sample_index].to_dict()}")
print(f"Actual disease status for sample {sample_index}: {true_label}")

# Predict using the best performing model (e.g., based on F1-Score or Accuracy)
# For this example, let's just use Random Forest
best_model_name = "Random Forest" # You might choose another based on your results
best_model = models[best_model_name]

predicted_label = best_model.predict(sample_features)[0]
predicted_proba = best_model.predict_proba(sample_features)[0][1] # Probability of being positive class

print(f"\nPrediction using {best_model_name}:")
print(f"  Predicted disease status: {predicted_label} (0: No Disease, 1: Disease)")
print(f"  Probability of disease: {predicted_proba:.4f}")

if predicted_label == true_label:
    print("  Prediction is CORRECT for this sample.")
else:
    print("  Prediction is INCORRECT for this sample.")

print("\n--- End of Disease Prediction Script ---")