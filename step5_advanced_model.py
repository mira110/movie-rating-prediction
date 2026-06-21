import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os

print("🚀 Starting Phase 5: Advanced Model with Genres...")

# 1. Load the CLEANED data
df = pd.read_csv("data/processed/cleaned_movies.csv", encoding='latin-1')

# 2. Clean up rows missing important data
df = df.dropna(subset=['year', 'duration', 'votes', 'rating', 'genre'])
print(f"Training advanced model on {len(df)} movies...")

# 3. Turn 'Genre' into numbers (1s and 0s)
# Example: "Action, Drama" becomes [Action=1, Drama=1, Comedy=0, ...]
print("Converting genres to numbers...")
df['genre_list'] = df['genre'].apply(lambda x: [g.strip() for g in x.split(',')])

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genre_list'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# 4. Combine our original numbers with the new genre numbers
X_numeric = df[['year', 'duration', 'votes']].reset_index(drop=True)
X_final = pd.concat([X_numeric, genre_df], axis=1)

y = df['rating']

# 5. Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 6. Train a powerful Gradient Boosting Model (Usually better than Random Forest)
from sklearn.ensemble import GradientBoostingRegressor
print("Training smarter model (This will take about 20-30 seconds)...")
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 7. Predict and Score
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\n" + "="*50)
print("🎓 ADVANCED MODEL RESULTS!")
print("="*50)
print(f"Average Mistake (MAE): {mae:.2f} stars")
print(f"Biggest Mistakes (RMSE): {rmse:.2f} stars")

if mae < 1.0:
    print("🔥 EXCELLENT! This model is highly accurate!")
else:
    print
# 8. SAVE THE MODEL (Crucial for your internship project!)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")
joblib.dump(mlb, "models/genre_encoder.pkl")
print("\n💾 Model saved successfully to 'models/best_model.pkl'!")
print("You can now use this model to predict ratings for new movies!")