

# 1. Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

# 2. Load Dataset
dataset_path = "/content/Crop_recommendation.csv"
data = pd.read_csv(dataset_path)

# 3. Preprocess Data
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target = "label"

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# 6. Save the Model
import joblib
joblib.dump(model, "/content/sample_data/model.pkl")

# 7. Recommendation Function
def recommend_crop(n, p, k, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame({
        "N": [n],
        "P": [p],
        "K": [k],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Example Usage
recommended_crop = recommend_crop(10, 42, 43, 25.5, 80, 6.5, 200)
print("Recommended Crop:", recommended_crop)
print("confusion Matrix",confusion_matrix(y_test,predictions))
