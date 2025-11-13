
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. Création d'un jeu de données synthétique
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(0, 2, 100)

# 3. Visualisation des données brutes
plt.scatter(X, y, color='blue')
plt.xlabel("Variable indépendante X")
plt.ylabel("Variable dépendante y")
plt.title("Données synthétiques pour la régression")
plt.show()

# 4. Séparation en données d'entraînement et de test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 5. Création et entraînement du modèle RNA (réseau de neurones multi-couches)
mlp = MLPRegressor(hidden_layer_sizes=(15, 10), activation='relu', max_iter=1500, random_state=1)
mlp.fit(X_train, y_train) # Ajustement du modèle sur les données d'entraînement

# 6. Prédiction sur les données de test
y_pred = mlp.predict(X_test)

# 7. Évaluation des performances du modèle
mse = mean_squared_error(y_test, y_pred) # Erreur quadratique moyenne
r2 = r2_score(y_test, y_pred) # Coefficient de détermination
print(f"Erreur quadratique moyenne (MSE) : {mse:.3f}")
print(f"Coefficient de détermination R2 : {r2:.3f}")

# 8. Visualisation de la comparaison réel vs prédit
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.scatter(X_test, y_pred, color='red', label='Prédictions du RNA')
plt.xlabel("Variable indépendante X")
plt.ylabel("Variable dépendante y")
plt.title("Comparaison données réelles et prédictions par RNA")
plt.legend()
plt.show()
