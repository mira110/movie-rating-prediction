import pandas as pd
import matplotlib.pyplot as plt
import os

# This tells the charts to show up nicely
plt.style.use('seaborn-v0_8-whitegrid')

# Create a folder to save our charts
os.makedirs("visualizations", exist_ok=True)

print("📊 Starting Phase 3: Creating Visualizations...")

# Load the CLEANED data (Notice we are now using the processed folder!)
df = pd.read_csv("data/processed/cleaned_movies.csv", encoding='latin-1')
print(f"Loaded {len(df)} cleaned movies.")

# CHART 1: Rating Distribution (How many movies got 5 stars, 6 stars, etc?)
plt.figure(figsize=(10, 5))
plt.hist(df['rating'].dropna(), bins=20, color='#4ECDC4', edgecolor='black')
plt.title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.savefig('visualizations/01_rating_distribution.png', bbox_inches='tight')
print("✅ Created: 01_rating_distribution.png")

# CHART 2: Movies per Year (Are more movies being made now?)
plt.figure(figsize=(12, 5))
year_counts = df['year'].value_counts().sort_index()
plt.plot(year_counts.index, year_counts.values, color='#FF6B6B', linewidth=2)
plt.title('Number of Movies Released Per Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.savefig('visualizations/02_movies_per_year.png', bbox_inches='tight')
print("✅ Created: 02_movies_per_year.png")

# CHART 3: Top 10 Genres (What type of movies are most common?)
plt.figure(figsize=(10, 6))
# Split genres because one movie can have multiple genres like "Action, Drama"
all_genres = []
for genres in df['genre'].dropna():
    all_genres.extend([g.strip() for g in genres.split(',')])

genre_counts = pd.Series(all_genres).value_counts().head(10)
plt.barh(genre_counts.index, genre_counts.values, color='#45B7D1')
plt.title('Top 10 Most Common Genres', fontsize=16, fontweight='bold')
plt.xlabel('Number of Movies', fontsize=12)
plt.gca().invert_yaxis() # Puts the biggest bar at the top
plt.savefig('visualizations/03_top_genres.png', bbox_inches='tight')
print("✅ Created: 03_top_genres.png")

print("\n🎉 All charts saved! Check your 'visualizations' folder on the left!")