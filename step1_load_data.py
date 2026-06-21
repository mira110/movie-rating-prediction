import pandas as pd
import os

print("Starting Phase 1: Data Loading...")

# This tells Python exactly where your file is
file_path = "data/raw/movies.csv"

# Check if the file actually exists in that spot
if os.path.exists(file_path):
    print("✅ File found! Loading data...")
    
    # Load the movie data into Python
    df = pd.read_csv(file_path, encoding='latin-1')
    
    print("\n" + "="*50)
    print("DATASET LOADED SUCCESSFULLY!")
    print("="*50)
    
    # Print how many rows (movies) and columns we have
    print(f"Total Movies: {len(df)}")
    print(f"Total Columns (Features): {len(df.columns)}")
    
    # Print the names of all the columns
    print("\n📋 Here are the column names:")
    for col in df.columns:
        print(f"  - {col}")
        
    # Show the first 3 movies in a nice table format
    print("\n👀 Here are the first 3 movies:")
    print(df.head(3).to_string())
    
    # Check how many empty/missing values are in each column
    print("\n❌ Missing values per column:")
    print(df.isnull().sum())
    
else:
    print("❌ ERROR: File not found!")
    print("Make sure 'movies.csv' is inside the 'data/raw/' folder.")