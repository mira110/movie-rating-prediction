🎬 Movie Rating Prediction Project
📋 Overview
This project uses Machine Learning to predict IMDB movie ratings based on features like genre, year, duration, and votes. It uses a Gradient Boosting Regressor model trained on Indian IMDB movie data.

🎯 Model Performance
Algorithm: Gradient Boosting Regressor
Average Mistake (MAE): ~0.81 stars
Features Used: Year, Duration, Votes, and One-Hot Encoded Genres
🛠️ Tech Stack
Language: Python
Data Manipulation: Pandas, NumPy
Machine Learning: Scikit-Learn
Visualization: Matplotlib, Seaborn
Web Framework: Streamlit
🚀 How to Run this Project
1. Install Requirements
Open your terminal in the project folder and run:

pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
2. Run the Web Application
In the terminal, run:
streamlit run app.py
A browser window will automatically open with the interactive prediction tool!

📁 Project Structure
data/raw/ - The original, messy dataset.
data/processed/ - The cleaned dataset ready for ML.
models/ - The saved AI brain (best_model.pkl).
visualizations/ - Charts showing data insights.
step1_load_data.py - Loading the raw data.
step2_clean_data.py - Cleaning parentheses, commas, and missing values.
step3_eda.py - Creating visual charts.
step4_first_model.py - Baseline model using only numbers.
step5_advanced_model.py - Advanced model using genres.
app.py - The interactive Streamlit web application.
👤 Author
Miraclin - Data Science Intern Candidate

4. Press **Ctrl + S** to save.

---

### 🏆 You are officially done with the core project!

**What you have to show the company now:**
1. ✅ Clean, well-organized folders.
2. ✅ A step-by-step data cleaning pipeline.
3. ✅ Beautiful charts in the `visualizations` folder.
4. ✅ A trained Machine Learning model.
5. ✅ A **working, interactive website** they can play with.
6. ✅ A professional README file explaining everything.

**How to present this to them:**
When you go to the interview or submit the project, simply tell them: *"I built an end-to-end machine learning pipeline. You can open the folder, run `streamlit run app.py`, and interact with the AI model I trained."* They will be incredibly impressed.

**Do you want to add anything else, or are you ready to package this up?**
🌐 Live Demo
Click here to test the live AI model: https://movie-rating-prediction-b3mke7wpjnr4tennbzzlxn.streamlit.app/
