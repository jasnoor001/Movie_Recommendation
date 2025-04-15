import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie rating data
data = {
    'User': ['Alice', 'Alice', 'Alice',
             'Bob', 'Bob', 'Bob',
             'Carol', 'Carol',
             'Dave', 'Eve', 'Frank'],
    'Movie': ['Titanic', 'Avatar', 'Inception',
              'Titanic', 'Inception', 'Interstellar',
              'Avatar', 'Inception',
              'Titanic', 'Avatar', 'Interstellar'],
    'Rating': [5, 3, 4,
               4, 5, 5,
               4, 5,
               2, 5, 4]
}


df = pd.DataFrame(data)

# Create user-movie matrix
user_movie_matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Transpose: movies as rows
movie_matrix = user_movie_matrix.T

# Compute cosine similarity between movies
movie_similarity = cosine_similarity(movie_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_matrix.index, columns=movie_matrix.index)

def recommend_movies_item_based(user_name, num_recommendations=3):
    if user_name not in user_movie_matrix.index:
        print(f"No data for user: {user_name}")
        return []

    # Get user's rated movies
    user_ratings = user_movie_matrix.loc[user_name]
    rated_movies = user_ratings[user_ratings > 0]

    scores = pd.Series(dtype=np.float64)

    # Score movies based on similarity to user's rated movies
    for movie, rating in rated_movies.items():
        similar_movies = movie_similarity_df[movie] * rating
        scores = scores.add(similar_movies, fill_value=0)

    # Remove movies already rated by the user
    scores = scores.drop(rated_movies.index, errors='ignore')

    # Sort and get top recommendations
    top_recommendations = scores.sort_values(ascending=False).head(num_recommendations)

    return list(top_recommendations.index)

# 🔍 Example usage
print("🎬 Recommendations for Alice:")
recommended = recommend_movies_item_based('Alice', num_recommendations=2)
for i, movie in enumerate(recommended, 1):
    print(f"{i}. {movie}")
