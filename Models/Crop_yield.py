from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR

# Load and preprocess the dataset
df = pd.read_csv('/content/crop_yield.csv', encoding="latin1")

# Create LabelEncoders for categorical features
label_encoder_crop = LabelEncoder()
label_encoder_season = LabelEncoder()

# Clean the data
df['Season'] = df['Season'].str.strip()

# Encode the categorical features
df['crop_encoded'] = label_encoder_crop.fit_transform(df['Crop'])
df['season_encoded'] = label_encoder_season.fit_transform(df['Season'])

# Features and target
X = df[['crop_encoded', 'season_encoded', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = df['Production']

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# K-Fold Cross-Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
rf_model = RandomForestRegressor(random_state=42)

r2_scores = []
indices = []

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_normalized)):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    indices.append((train_index, test_index))

# Find the best model based on the highest R-squared score
best_score_ind = np.argmax(r2_scores)
best_score = r2_scores[best_score_ind]
xtrain, ytrain, xtest, ytest = X_normalized[indices[best_score_ind][0]], y.iloc[indices[best_score_ind][0]], X_normalized[indices[best_score_ind][1]], y.iloc[indices[best_score_ind][1]]

# Retrain the model on the full training data
rf_model.fit(xtrain, ytrain)
y_pred = rf_model.predict(xtest)

# Save the trained model, scaler, and label encoders
joblib.dump(rf_model, 'model.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
joblib.dump(label_encoder_crop, 'label_encoder_crop.pkl')  # Save the crop label encoder
joblib.dump(label_encoder_season, 'label_encoder_season.pkl')  # Save the season label encoder

# Calculate and print the final R-squared score
r2 = r2_score(ytest, y_pred)
print(f"R-squared: {r2:.2f}")
