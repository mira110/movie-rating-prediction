import pandas as pd

print("🧹 Starting Phase 2: Data Cleaning...")

# 1. Load the raw data
df = pd.read_csv("data/raw/movies.csv", encoding='latin-1')
print(f"Started with {len(df)} rows of messy data.")

# 2. Make all column names lowercase
df.columns = df.columns.str.strip().str.lower()

# 3. Clean 'year' (remove parentheses like "(2019)" -> "2019")
if 'year' in df.columns:
    df['year'] = df['year'].astype(str).str.replace('(', '', regex=False)
    df['year'] = df['year'].str.replace(')', '', regex=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

# 4. Clean 'duration' (remove " min" -> just the number)
if 'duration' in df.columns:
    df['duration'] = df['duration'].astype(str).str.replace(' min', '', regex=False)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

# 5. Clean 'votes' (remove commas like "1,00,000" -> "100000")
if 'votes' in df.columns:
    df['votes'] = df['votes'].astype(str).str.replace(',', '', regex=False)
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# 6. Remove rows where 'rating' is missing (we need ratings to train the model!)
if 'rating' in df.columns:
    df = df.dropna(subset=['rating'])

print(f"After cleaning, we have {len(df)} ready rows.")

# 7. Save the clean data to a NEW file
df.to_csv("data/processed/cleaned_movies.csv", index=False)

print("\n✅ SUCCESS! Clean data saved to: data/processed/cleaned_movies.csv")