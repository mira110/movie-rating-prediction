import pandas as pd
import os

print("Starting Phase 1: Data Loading...")

file_path = "data/raw/movies.csv"

# Check if file exists
if os.path.exists(file_path):
    print("✅ File found! Loading data...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("DATASET LOADED SUCCESSFULLY!")
    print("="*50)
    
    # Print how many rows and columns
    print(f"Total Movies: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    print("\n📋 Here are the column names:")
    for col in df.columns:
        print(f"  - {col}")
        
    print("\n👀 Here are the first 3 movies:")
    print(df.head(3).to_string())
    
else:
    print("❌ ERROR: File not found!")
    print("Make sure 'movies.csv' is inside the 'data/raw/' folder.")