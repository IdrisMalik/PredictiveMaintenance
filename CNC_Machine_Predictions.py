import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime, timedelta

# Load your dataset
data = pd.read_csv('cnc_machine_dataset.csv')  # Replace with your dataset file

# Preprocessing
# Assuming your dataset has columns like 'sensor_1', 'sensor_2', ..., 'timestamp', 'failure'
# You may need to adapt this based on your actual dataset

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Feature engineering
# Calculate time differences between consecutive readings
data['time_diff'] = data['timestamp'].diff().dt.total_seconds().fillna(0)

# Assuming 'failure' column indicates machine failure (1) or not (0)
X = data.drop(['timestamp', 'failure'], axis=1)
y = data['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict maintenance alerts and lifetime
latest_reading = data.iloc[-1]
current_time = latest_reading['timestamp']
next_reading_time = current_time + timedelta(minutes=15)  # Assuming readings are taken every 15 minutes

next_reading = latest_reading.drop(['timestamp', 'failure'])
next_reading['time_diff'] = (next_reading_time - current_time).total_seconds()

predicted_failure = model.predict([next_reading])[0]
maintenance_alert = "Maintenance needed" if predicted_failure == 1 else "No maintenance needed"

# Predict remaining lifetime
# You would need more advanced methods to estimate remaining lifetime accurately
remaining_lifetime = model.predict_proba([next_reading])[0][1]  # Probability of failure

print(f"Maintenance Alert: {maintenance_alert}")
print(f"Predicted Remaining Lifetime: {remaining_lifetime:.2%} probability of failure")