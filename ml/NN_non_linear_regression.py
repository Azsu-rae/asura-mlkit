
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Génération des données non linéaires
np.random.seed(42)
x_data = np.linspace(-10, 10, 1000).reshape(-1, 1)
noise = 0.1 * np.random.normal(size=x_data.shape[0])
y_data = 0.1 * x_data.flatten() * np.cos(x_data.flatten()) + noise

# Visualisation des données
plt.scatter(x_data, y_data, s=10, color='blue')
plt.xlabel('Variable indépendante x')
plt.ylabel('Variable dépendante y')
plt.title('Données synthétiques NON LINÉAIRES pour régression')
plt.show()

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

# Modèle RNA
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000,
random_state=1)
mlp.fit(X_train, y_train)

# Prédiction
y_pred = mlp.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erreur quadratique moyenne (MSE) : {mse:.4f}")
print(f"Coefficient de détermination R2 : {r2:.4f}")

# Visualisation résultats
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.scatter(X_test, y_pred, color='red', label='Prédictions RNA')
plt.xlabel('Variable indépendante x')
plt.ylabel('Variable dépendante y')
plt.title('Comparaison réel vs prédit par RNA')
plt.legend()
plt.show()
