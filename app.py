```
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import gradio as gr

# Load the data
df = pd.read_csv('Maternal Health Risk Data Set.csv')

# Separate features and target variable
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=skf, scoring='accuracy')

# Print average accuracy across folds
print(f'Accuracy: {cv_scores.mean():.2f}')

# Fit the model on the entire dataset
rf_classifier.fit(X, y)

# Predict function for Gradio
def predict(Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate):
    input_data = scaler.transform([[Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]])
    prediction = rf_classifier.predict(input_data)
    return label_encoder.inverse_transform(prediction)[0]

import gradio as gr
inputs = [
    gr.Slider(label="Age", info="Enter your age",maximum=74),
    gr.Slider(minimum=70, maximum=160, step=1, label="Systolic Blood Pressure", info="Range: 70-160 mmHg"),
    gr.Slider(minimum=49, maximum=100, step=1, label="Diastolic Blood Pressure", info="Range: 49-100 mmHg"),
    gr.Slider(minimum=6, maximum=19, step=0.1, label="Blood Sugar (mmol/L)", info="Range: 6-19 mmol/L"),
    gr.Slider(minimum=98, maximum=103, step=0.1, label="Body Temperature", info="Range: 98-103 °F"),
    gr.Slider(minimum=60, maximum=90, step=1, label="Heart Rate", info="Range: 60-90 bpm")
]
outputs = gr.Label()
iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Maternal Health Risk Predictor",
    description=(
        "Empower your health decisions with the nurturing guidance of maternal wisdom, "
        "enhanced by advanced machine learning and data science. Uncover predictive insights "
        "using sacred health indicators, custom-tailored for precise maternal health risk assessment. "
        "Please note the input limits: Age (in years), Systolic BP (70-160 mmHg), Diastolic BP(49-100 mmHg), "
        "Blood Sugar (6-19 mmol/L),Body Temperature (98-103°F),Heart Rate(60-90 bpm)."
    ),
    
)

iface.launch()
```
