# 🫀 Heart Disease Prediction using k-NN Classifier

## 📘 Project Overview

This project focuses on predicting the likelihood of **heart disease** in patients based on clinical parameters using **Machine Learning techniques**.  
The dataset is processed, analyzed, and modeled using a **K-Nearest Neighbors (k-NN)** classifier with **hyperparameter tuning** to improve performance.

The goal is to explore the relationships between patient characteristics and heart disease presence, clean and visualize the dataset, and build a predictive model that generalizes well to unseen data.

---

## 🧰 Technologies and Libraries Used

- **Python 3.x**
- **pandas** → Data manipulation and preprocessing  
- **numpy** → Numerical computations  
- **matplotlib** & **seaborn** → Data visualization  
- **scikit-learn** → Machine learning, preprocessing, and model evaluation

---

## 📂 Dataset

- **Filename:** `heart_disease_prediction.csv`
- **Number of features:** 12  
- **Number of observations:** 918  

### 🧾 Columns Overview

| Feature | Description | Type |
|----------|-------------|------|
| Age | Patient's age | int |
| Sex | Male or Female | object |
| ChestPainType | Type of chest pain (ATA, NAP, ASY, etc.) | object |
| RestingBP | Resting blood pressure (mm Hg) | int |
| Cholesterol | Serum cholesterol (mg/dl) | int |
| FastingBS | Fasting blood sugar (>120 mg/dl) | int |
| RestingECG | Resting electrocardiogram results | object |
| MaxHR | Maximum heart rate achieved | int |
| ExerciseAngina | Exercise-induced angina (Y/N) | object |
| Oldpeak | ST depression induced by exercise | float |
| ST_Slope | Slope of the peak exercise ST segment | object |
| HeartDisease | Target variable (1 = presence, 0 = absence) | int |

---

## 🔍 Data Exploration & Cleaning

### Key Insights
- **Average age** of patients: ~53.5 years  
- **Abnormally high cholesterol** values (up to 603 mg/dl) identified  
- **Missing or invalid data:**  
  - `RestingBP` had 1 value equal to 0 (invalid)  
  - `Cholesterol` had 172 values equal to 0 (replaced with median values by group)  
- **No missing (NaN)** values detected  

### Outlier Treatment & Imputation
Invalid zero values were replaced with **median values** grouped by:
`[HeartDisease, Age, Sex]`

---

## 📊 Data Visualization

#### 1. Distribution of Categorical Variables
- Most patients were **male**  
- The **ASY** chest pain type was most common among those with heart disease  
- Patients with **exercise-induced angina (Y)** or **ST_Slope_Flat** showed higher risk

#### 2. Correlation Heatmap
A correlation matrix was used to identify the most relevant features:
- Strongly correlated features with `HeartDisease`:  
  **Oldpeak, MaxHR, ExerciseAngina_Y, ST_Slope_Flat, ST_Slope_Up, Sex_M**

---

## 🧮 Feature Engineering

Categorical features were converted using **one-hot encoding** (`pd.get_dummies`) with `drop_first=True` to avoid dummy variable traps.

### Selected Features
```python
Features = ['Oldpeak', 'MaxHR', 'Sex_M', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']
Target = 'HeartDisease'


## 🤖 Model Building

### 🧩 Step 1: Individual Feature Performance

Each feature was trained using **k-NN (k=5)** independently.

| Feature | Accuracy |
|----------|-----------|
| Oldpeak | 0.6957 |
| MaxHR | 0.6141 |
| Sex_M | 0.4185 |
| ExerciseAngina_Y | 0.6576 |
| ST_Slope_Flat | 0.7500 |
| **ST_Slope_Up** | **0.7989** |

➡️ **Best single feature:** `ST_Slope_Up` with accuracy **0.7989**

---

### ⚖️ Step 2: Model with All Features (Scaled)

After applying **MinMaxScaler**, the model achieved:
Accuracy with all features: 0.7880


Although slightly lower than the best single feature, using all features increased robustness and captured more data complexity.

---

## ⚙️ Hyperparameter Tuning

Grid search was applied to optimize model parameters.

### 🔧 Grid Search Parameters
```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
🏁 Best Results
Best Score: 0.8556
Best Parameters: {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'uniform'}


🧪 Final Test Accuracy
Test Set Accuracy: 0.7826

🧠 Results Summary
Step	Model	Accuracy	Notes
Individual Feature (ST_Slope_Up)	k-NN	0.7989	Best single feature
All Features (Scaled)	k-NN	0.7880	Slightly lower but more robust
After GridSearchCV	k-NN (optimized)	0.8556 (CV), 0.7826 (test)	Best configuration
🩺 Key Conclusions

ST_Slope_Up is the most influential feature.

The k-NN algorithm provides good interpretability and competitive performance after scaling and tuning.

An accuracy of ~78% indicates a moderately strong predictive power — suitable for exploratory analysis, but not yet for clinical deployment.

🚀 Future Improvements

🔹 Apply feature selection using Recursive Feature Elimination (RFE)

🔹 Try advanced models: RandomForest, XGBoost, or Logistic Regression

🔹 Perform cross-validation with more folds for robustness

🔹 Add model interpretability using SHAP or LIME

🔹 Deploy the model using Streamlit for interactive prediction

🧾 Author

Developed by:Asma Mestaysser 

🏷️ License

This project is open-source and available under the MIT License.

💡 Example Usage
# Clone this repository
git clone https://github.com/yourusername/heart-disease-prediction.git

# Navigate to the project directory
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
python heart_disease_prediction.py

🫀 “Prevention starts with prediction.”
