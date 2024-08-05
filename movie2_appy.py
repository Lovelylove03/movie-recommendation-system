import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split

# Streamlit title
st.title('Movie Recommendation System')

# File path
file_path = 'movies_step2.csv'

# Read the CSV file
df_movies = pd.read_csv(file_path, sep=',')

# Convert 'runtimeMinutes' column to int
df_movies['runtimeMinutes'] = pd.to_numeric(df_movies['runtimeMinutes'], errors='coerce')
df_movies['runtimeMinutes'] = df_movies['runtimeMinutes'].fillna(0).astype(int)

# Train test split
X = df_movies.drop(columns=['tconst', 'titleType', 'title', 'language'])
y = df_movies['title']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Display dataset information
st.write("The length of the initial dataset is :", len(X))
st.write("The length of the train dataset is   :", len(X_train))
st.write("The length of the test dataset is    :", len(X_test))

# Train KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
st.write("KNeighborsClassifier model:", model)
st.write("Train score:", model.score(X_train, y_train))
st.write("Test score:", model.score(X_test, y_test))

# Evaluate different hyperparameters
for neighbors in range(2, 6):
    for weight in ["uniform", "distance"]:
        model = KNeighborsClassifier(n_neighbors=neighbors, weights=weight).fit(X_train, y_train)
        st.write(f"For {neighbors} neighbors and weight={weight}: train score {model.score(X_train, y_train)}, test score: {model.score(X_test, y_test)}")

# Train NearestNeighbors model
modelNN = NearestNeighbors(n_neighbors=3)
modelNN.fit(X)

# Movie recommendation
list_movie = ['Kate et Léopold']
for movie in list_movie:
    neighbors = modelNN.kneighbors(df_movies.loc[df_movies['title'] == movie, X.columns])
    closest_movies_indices = neighbors[1][0]
    closest_movies = df_movies['title'].iloc[closest_movies_indices]
    st.write(f"Recommendations for movie {movie}:")
    st.write("Closest Movies:", list(closest_movies))
    st.write("Respective distances:", neighbors[0][0])
numpy
pandas
seaborn
scikit-learn
streamlit
env/
__pycache__/
*.pyc
*.pyo
# Movie Recommendation System

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
