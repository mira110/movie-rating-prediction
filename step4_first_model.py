import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

print("🤖 Starting Phase 4: Training First Machine Learning Model...")

# 1. Load the CLEANED data
df = pd.read_csv("data/processed/cleaned_movies.csv", encoding='latin-1')

# 2. Remove any rows that are missing our number columns
df = df.dropna(subset=['year', 'duration', 'votes', 'rating'])

print(f"Training model on {len(df)} movies...")

# 3. Pick our features (X) and what we want to predict (y)
X = df[['year', 'duration', 'votes']]  # The inputs
y = df['rating']                       # The output we want to guess

# 4. Split data into "Practice" (Train) and "Exam" (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create the Brain (Random Forest Model)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 6. Let the Brain study the practice data
print("Studying the data (This might take 10 seconds)...")
model.fit(X_train, y_train)

# 7. Let the Brain take the exam
predictions = model.predict(X_test)

# 8. Grade the exam
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\n" + "="*50)
print("🎓 MODEL EXAM RESULTS!")
print("="*50)
print(f"Average Mistake (MAE): {mae:.2f} stars")
print(f"Biggest Mistakes (RMSE): {rmse:.2f} stars")
print("\n💡 What does this mean?")
print(f"If the model guesses a movie is a 7.0, it's usually only off by {mae:.2f} stars!")