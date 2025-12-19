"""
TP3 - Exercice 3: Segmentation en régions (Croissance de régions)
Implémentation de l'algorithme de croissance de régions (Region Growing)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def region_growing(img, seed, threshold=15):
    """
    Segmentation par croissance de région à partir d'un point germe (seed).
    
    Args:
        img: Image en niveaux de gris
        seed: Tuple (x, y) du point de départ
        threshold: Seuil de tolérance pour le critère d'homogénéité
        
    Returns:
        mask: Image binaire correspondant à la région segmentée
    """
    rows, cols = img.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=np.bool_)
    
    # Point de départ (attention: seed est (x,y), image est [y,x])
    seed_x, seed_y = seed
    
    # Vérification que le seed est dans l'image
    if not (0 <= seed_x < cols and 0 <= seed_y < rows):
        raise ValueError(f"Le point germe {seed} est hors de l'image")
        
    seed_value = float(img[seed_y, seed_x])
    
    # File d'attente pour les pixels à traiter
    queue = deque()
    queue.append((seed_y, seed_x))
    visited[seed_y, seed_x] = True
    mask[seed_y, seed_x] = 255
    
    # Directions (8 voisins)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    while queue:
        y, x = queue.popleft()
        
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            
            # Vérifier les limites et si déjà visité
            if 0 <= ny < rows and 0 <= nx < cols and not visited[ny, nx]:
                # Critère d'homogénéité: écart absolu avec la valeur du seed
                # On pourrait aussi utiliser la moyenne courante de la région
                if abs(float(img[ny, nx]) - seed_value) < threshold:
                    visited[ny, nx] = True
                    mask[ny, nx] = 255
                    queue.append((ny, nx))
                    
    return mask

def sequential_region_growing(img, threshold=15):
    """
    Segmentation séquentielle automatique de toutes les régions.
    
    Args:
        img: Image en niveaux de gris
        threshold: Seuil de tolérance
        
    Returns:
        result_img: Image où chaque région est coloriée avec sa valeur moyenne
    """
    rows, cols = img.shape
    visited = np.zeros((rows, cols), dtype=np.bool_)
    result_img = np.zeros((rows, cols), dtype=np.uint8)
    
    # Parcourir tous les pixels
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                # Démarrer une nouvelle croissance de région
                region_pixels = []
                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                region_pixels.append(img[i, j])
                
                seed_value = float(img[i, j])
                
                # Directions (8 voisins)
                neighbors = [(-1, -1), (-1, 0), (-1, 1),
                             (0, -1),           (0, 1),
                             (1, -1),  (1, 0),  (1, 1)]
                
                # Pour reconstruire la région dans result_img après
                current_region_indices = [(i, j)]
                
                while queue:
                    y, x = queue.popleft()
                    
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        
                        if 0 <= ny < rows and 0 <= nx < cols and not visited[ny, nx]:
                            if abs(float(img[ny, nx]) - seed_value) < threshold:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                                region_pixels.append(img[ny, nx])
                                current_region_indices.append((ny, nx))
                
                # Calculer la moyenne de la région
                if region_pixels:
                    mean_val = int(np.mean(region_pixels))
                    # Remplir la région avec la valeur moyenne
                    for ry, rx in current_region_indices:
                        result_img[ry, rx] = mean_val
                        
    return result_img

def main():
    # Paramètres
    image_path = "Images/Cerveau_AVC1.JPG"
    seed_point = (260, 140) # (x, y)
    threshold = 15
    
    print(f"TP3 Exo 3 - Segmentation par région")
    print(f"Chargement de l'image: {image_path}")
    
    # 1. Charger l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur: Impossible de trouver l'image '{image_path}'")
        print("Tentative avec une image de remplacement (compteur.jpg) si disponible...")
        # Fallback pour le test si l'image demandée n'existe pas
        img = cv2.imread("Images/compteur.jpg", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erreur: Aucune image trouvée. Veuillez vérifier le dossier Images.")
            return
        else:
            print("Image de remplacement chargée.")
            seed_point = (img.shape[1]//2, img.shape[0]//2) # Seed au centre par défaut
    
    print(f"Image chargée: {img.shape}")
    
    # Lissage Gaussien comme demandé
    print("Application du lissage Gaussien...")
    img_smooth = cv2.GaussianBlur(img, (5, 5), 4)
    
    # 2. Croissance de région unique
    print(f"Exécution de region_growing avec seed={seed_point} et seuil={threshold}...")
    try:
        mask_single = region_growing(img_smooth, seed_point, threshold)
    except Exception as e:
        print(f"Erreur lors de region_growing: {e}")
        mask_single = np.zeros_like(img_smooth)

    # 3. Croissance de régions séquentielle (Automatique)
    # Note: Cela peut prendre du temps sur une grande image en Python pur
    print("Exécution de sequential_region_growing (cela peut prendre quelques secondes)...")
    # On peut réduire la taille de l'image pour le test si c'est trop lent
    # img_small = cv2.resize(img_smooth, (0,0), fx=0.5, fy=0.5)
    full_segmentation = sequential_region_growing(img_smooth, threshold)
    
    # Affichage
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img_smooth, cmap='gray')
    plt.title('Image lissée')
    # Marquer le seed point
    plt.plot(seed_point[0], seed_point[1], 'r+', markersize=10)
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask_single, cmap='gray')
    plt.title(f'Région unique (Seed: {seed_point})')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(full_segmentation, cmap='jet') # Jet permet de mieux visualiser les différences de gris
    plt.title('Segmentation séquentielle complète')
    plt.axis('off')
    
    output_file = 'CV_TP3/segmentation_regions.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Résultats sauvegardés dans {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
