import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
file_path = r"C:movies_step2 - movies_step2.csv"
 # Lire le fichier CSV
df_movies = pd.read_csv(file_path, sep=',')
 #Convertir la colonne 'runtimeMinutes' en int
# 1. Remplacer les valeurs non numériques et les chaînes vides par NaN
df_movies['runtimeMinutes'] = pd.to_numeric(df_movies['runtimeMinutes'], errors='coerce')

# 2. Remplacer les valeurs NaN par 0 (ou toute autre valeur par défaut)
df_movies['runtimeMinutes'] = df_movies['runtimeMinutes'].fillna(0).astype(int)
# Train test split

X = df_movies.drop(columns=['tconst','titleType', 'title', 'language'])
y = df_movies['title']
# Your code here

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42, )
print("The length of the initial dataset is :", len(X))
print("The length of the train dataset is   :", len(X_train))
print("The length of the test dataset is    :", len(X_test))
model =  KNeighborsClassifier()
model.fit(X_train, y_train)
print(model)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
for neighbors in range(2,6):
  for weight in ["uniform" , "distance"]:
    model =  KNeighborsClassifier(n_neighbors = neighbors , weights = weight).fit(X_train, y_train)
    print("For ", neighbors, "neighbors and weight=", weight,
          ": train score", model.score(X_train, y_train),
          "and test score:", model.score(X_test, y_test))
X = df_movies.drop(columns=['tconst','titleType', 'title', 'language'])
y = df_movies['title']
modelNN = NearestNeighbors(n_neighbors=3)
modelNN.fit(X_train, y_train)
list_movie = ['Kate et Léopold']
for movie in list_movie:
    # for a movie, pass only same columns as X that our model needs
    neighbors = modelNN.kneighbors(df_movies.loc[df_movies['title'] == movie, X.columns])
    print(f"Recommandations for movie {movie} :")
    # find row number (not row name) from the nearest neighbors into dataframe on which the model was fitted
    closest_pok_ind = neighbors[1][0]
    closest_pok = df_movies['title'].iloc[closest_pok_ind]
    print("Closest Moviess : ", list(closest_pok))
    print("Respectives distances : ", neighbors[0][0])
    print()
