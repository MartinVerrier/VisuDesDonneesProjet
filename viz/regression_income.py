# Importez les librairies nécessaires
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
import pandas as pd
import numpy as np

# Chargez les données
data = pd.read_csv("/home/martin/Bureau/Cours/INFO5/VisuDonnées/Projet/VisuDesDonneesProjet/data/income_averages.tsv", sep='\t')

# Séparez les variables indépendantes et dépendantes
X = data[["country", "year"]]
y = data["average"]

# Utilisez one-hot encoding pour transformer les variables catégorielles en variables binaires
encoder = OneHotEncoder(handle_unknown = 'ignore')
X = encoder.fit_transform(X)

# Séparez les données en un jeu d'entraînement et un jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez un objet de régression multiple
reg = LinearRegression()

# Entraînez le modèle sur les données d'entraînement
reg.fit(X_train, y_train)

# Utilisez le modèle pour faire des prévisions sur les données de test
y_pred = reg.predict(X_test)

# Évaluez la performance du modèle en utilisant des métriques d'évaluation appropriées
print("R²: ", r2_score(y_test, y_pred))

# Calcul de MAE
print("MAE : ", mean_absolute_error(y_test, y_pred))

# Calcul de MSE
print("MSE : ", mean_squared_error(y_test, y_pred))

# Calcul de RMSE
print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_pred)))

# # 1914 - 2017

# # Créez un dataframe vide pour stocker les nouvelles données
# new_data = pd.DataFrame(columns=["country", "year", "average", "is_estimate"])

# # Bouclez sur chaque pays
# for country in data["country"].unique():
#     # Bouclez sur chaque année entre 1914 et 2017
#     for year in range(1914, 2018):
#         # Vérifiez si l'année et le pays existent déjà dans les données
#         if data.query("year == @year and country == @country").empty == True :
#             # Utilisez le modèle pour prédire la valeur de l'indice Gini pour cette année et ce pays
#             new_data_temp = pd.DataFrame({'country': [country], 'year': [year]})
#             encoded_new_data = encoder.transform(new_data_temp[["country", "year"]])
#             gini_pred = reg.predict(encoded_new_data)[0]
#             # Ajoutez les données prédites à new_data
#             new_data = new_data.append(pd.DataFrame({"country": country, "year": year, "average": gini_pred, "is_estimate": True}, index=[0]), ignore_index=False)

# # Concatenez les nouvelles données avec les anciennes
# data = pd.concat([data, new_data], axis=0)

# # Triez les données par pays et par date
# data = data.sort_values(by=['country', 'year'], ascending=[True, False])

# # Enregistrez les données mises à jour dans un nouveau fichier
# data.to_csv("/home/martin/Bureau/Cours/INFO5/VisuDonnées/Projet/VisuDesDonneesProjet/data/new_income_averages.tsv", sep='\t')