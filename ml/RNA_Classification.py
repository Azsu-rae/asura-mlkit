
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
#print(X[y == label, 0])
#X[y == label, 1]

plt.figure(figsize=(7,5))
for label, color, name in zip([0, 1], ['blue', 'red'], ['Classe 0', 'Classe 1']):
    plt.scatter(X[y == label, 0], X[y == label, 1], c=color, label=name, edgecolors='k')

plt.xlabel('Coordonnée X1')
plt.ylabel('Coordonnée X2')
plt.title('Données synthétiques générées - Classes réelles')
plt.legend()
plt.show()

# 3. Séparer les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

# 4. Création et entraînement du MLP
model = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=350, random_state=42)
model.fit(X_train, y_train)

# 5. Prédiction sur le test
y_pred = model.predict(X_test)

# 6. Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur le jeu de test : {accuracy:.2f}")

# 7. Visualisation des prédictions sur le jeu de test avec légende
plt.figure(figsize=(7,5))
for label, color, name in zip([0, 1], ['blue', 'red'], ['Classe prédite 0', 'Classe prédite 1']):
    plt.scatter(
        X_test[y_pred == label, 0],
        X_test[y_pred == label, 1],
        c=color,
        label=name,
        edgecolors='k'
    )

plt.xlabel('Coordonnée X1')
plt.ylabel('Coordonnée X2')
plt.title('Prédictions de classification sur le jeu de test')
plt.legend()
plt.show()
