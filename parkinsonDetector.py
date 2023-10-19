import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, render_template, request
import io
import base64


# Read the data
df = pd.read_csv('parkinsons.data')

# Get the features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Scale the features to between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train the model
model = XGBClassifier()
model.fit(x_train, y_train)

# Calculate the accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Save the trained model
model_filename = 'parkinsons_xgboost_model.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved as '{model_filename}'")

# Load the trained model
model = joblib.load(model_filename)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            return render_template('index.html', error='Please select a file.')

        data = pd.read_csv(file)
        features = data.drop(columns=['name'])
        scaler = MinMaxScaler((-1, 1))
        scaled_features = scaler.fit_transform(features)
        predictions = model.predict(scaled_features)
        data['Prediction'] = predictions

        # Convert DataFrame to CSV and then to base64
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        encoded_csv = base64.b64encode(csv_data.encode()).decode()

        return render_template('index.html', predictions=encoded_csv)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
