import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("Current working directory:", os.getcwd())
# Load the dataset
file_path = os.path.join(os.getcwd(), 'data/raw/volumes_atr_cyclists_shortterm.csv')
#file_path = '../../data/raw/volumes_atr_cyclists_shortterm.csv'
df = pd.read_csv(file_path)

# Convert datetime strings to datetime objects
df['datetime_bin_start'] = pd.to_datetime(df['datetime_bin_start'])

# Extract additional time-based features
df['hour'] = df['datetime_bin_start'].dt.hour
df['day_of_week'] = df['datetime_bin_start'].dt.dayofweek
df['month'] = df['datetime_bin_start'].dt.month

# Define features
numeric_features = ['hour', 'day_of_week', 'month', 'daily_temperature', 'daily_precipitation']
categorical_features = ['direction', 'location', 'class_type']

# Combine all features
features = numeric_features + categorical_features
X = df[features]
y = df['volume']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'models/final_model.joblib')
print("Model saved to 'models/final_model.joblib'")

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")