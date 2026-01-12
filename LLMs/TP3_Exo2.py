"""
TP3 - Exercice 2: Segmentation d'image en contours
Implémentation des détecteurs de contours: Roberts, Prewitt, Sobel, et Canny
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def quantize_orientation(theta):
    """
    Quantifie l'orientation du gradient en 4 directions (0°, 45°, 90°, 135°)
    
    Args:
        theta: Orientation en radians
    
    Returns:
        Angle quantifié (0, 45, 90, 135)
    """
    # Convertir en degrés et normaliser entre 0 et 180
    angle = np.abs(np.degrees(theta)) % 180
    
    # Quantifier en 4 directions
    if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
        return 0
    elif 22.5 <= angle < 67.5:
        return 45
    elif 67.5 <= angle < 112.5:
        return 90
    else:  # 112.5 <= angle < 157.5
        return 135


def non_maximum_suppression(gradient_magnitude, gradient_orientation):
    """
    Suppression des non-maximum locaux (NMS)
    Affine les contours en ne conservant que les pixels qui sont des maximums locaux
    dans la direction du gradient.
    
    Args:
        gradient_magnitude: Image d'amplitude du gradient
        gradient_orientation: Image d'orientation du gradient (en radians)
    
    Returns:
        Image avec contours affinés
    """
    rows, cols = gradient_magnitude.shape
    nms = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Quantifier l'orientation
            angle = quantize_orientation(gradient_orientation[i, j])
            
            # Déterminer les deux voisins dans la direction du gradient
            if angle == 0:  # Horizontal
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
            elif angle == 45:  # Diagonale /
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            elif angle == 90:  # Vertical
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            else:  # angle == 135, Diagonale \
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            
            # Conserver uniquement si c'est un maximum local
            if gradient_magnitude[i, j] >= max(neighbors):
                nms[i, j] = gradient_magnitude[i, j]
    
    return nms


def hysteresis_thresholding(image, low_threshold, high_threshold):
    """
    Seuillage par hystérésis
    Applique deux seuils pour classer les pixels en "contours forts", "contours faibles" et "fond"
    Conserve les contours faibles uniquement s'ils sont connectés à un contour fort
    
    Args:
        image: Image d'entrée (gradients après NMS)
        low_threshold: Seuil bas
        high_threshold: Seuil haut
    
    Returns:
        Image binaire des contours
    """
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    
    # Classification des pixels
    strong = 255
    weak = 75
    
    # Pixels forts et faibles
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    # Connecter les contours faibles aux contours forts
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if result[i, j] == weak:
                # Vérifier le voisinage 3x3
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    
    # Binariser: conserver uniquement les contours forts
    result[result == weak] = 0
    
    return result


def sobel_edge_detection(im, sb=30, sh=70):
    """
    Détection de contours avec l'opérateur de Sobel
    
    Args:
        im: Image en niveaux de gris
        sb: Seuil bas pour hystérésis
        sh: Seuil haut pour hystérésis
    
    Returns:
        edges: Image binaire des contours
        gradient_mag: Amplitude du gradient
        gradient_orient: Orientation du gradient
    """
    # Masques de Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    # Appliquer les convolutions
    Gx = cv2.filter2D(im.astype(np.float32), -1, sobel_x)
    Gy = cv2.filter2D(im.astype(np.float32), -1, sobel_y)
    
    # Calculer l'amplitude du gradient
    gradient_mag = np.sqrt(Gx**2 + Gy**2)
    
    # Calculer l'orientation du gradient
    gradient_orient = np.arctan2(Gy, Gx)
    
    # Suppression des non-maximum locaux
    nms_result = non_maximum_suppression(gradient_mag, gradient_orient)
    
    # Seuillage par hystérésis
    edges = hysteresis_thresholding(nms_result, sb, sh)
    
    return edges, gradient_mag, gradient_orient


def robert_edge_detection(im, sb=30, sh=70):
    """
    Détection de contours avec l'opérateur de Roberts
    
    Args:
        im: Image en niveaux de gris
        sb: Seuil bas pour hystérésis
        sh: Seuil haut pour hystérésis
    
    Returns:
        edges: Image binaire des contours
        gradient_mag: Amplitude du gradient
        gradient_orient: Orientation du gradient
    """
    # Masques de Roberts
    roberts_x = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    
    roberts_y = np.array([[0, 1],
                          [-1, 0]], dtype=np.float32)
    
    # Appliquer les convolutions
    Gx = cv2.filter2D(im.astype(np.float32), -1, roberts_x)
    Gy = cv2.filter2D(im.astype(np.float32), -1, roberts_y)
    
    # Calculer l'amplitude du gradient
    gradient_mag = np.sqrt(Gx**2 + Gy**2)
    
    # Calculer l'orientation du gradient
    gradient_orient = np.arctan2(Gy, Gx)
    
    # Suppression des non-maximum locaux
    nms_result = non_maximum_suppression(gradient_mag, gradient_orient)
    
    # Seuillage par hystérésis
    edges = hysteresis_thresholding(nms_result, sb, sh)
    
    return edges, gradient_mag, gradient_orient


def prewitt_edge_detection(im, sb=30, sh=70):
    """
    Détection de contours avec l'opérateur de Prewitt
    
    Args:
        im: Image en niveaux de gris
        sb: Seuil bas pour hystérésis
        sh: Seuil haut pour hystérésis
    
    Returns:
        edges: Image binaire des contours
        gradient_mag: Amplitude du gradient
        gradient_orient: Orientation du gradient
    """
    # Masques de Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float32)
    
    # Appliquer les convolutions
    Gx = cv2.filter2D(im.astype(np.float32), -1, prewitt_x)
    Gy = cv2.filter2D(im.astype(np.float32), -1, prewitt_y)
    
    # Calculer l'amplitude du gradient
    gradient_mag = np.sqrt(Gx**2 + Gy**2)
    
    # Calculer l'orientation du gradient
    gradient_orient = np.arctan2(Gy, Gx)
    
    # Suppression des non-maximum locaux
    nms_result = non_maximum_suppression(gradient_mag, gradient_orient)
    
    # Seuillage par hystérésis
    edges = hysteresis_thresholding(nms_result, sb, sh)
    
    return edges, gradient_mag, gradient_orient


def test_edge_detectors(image_path):
    """
    Teste et compare les différents détecteurs de contours
    
    Args:
        image_path: Chemin vers l'image à analyser
    """
    # Charger l'image en niveaux de gris
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if im is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return
    
    print(f"Image chargée: {image_path}, taille: {im.shape}")
    
    # Appliquer un lissage gaussien (prétraitement recommandé)
    im_smooth = cv2.GaussianBlur(im, (5, 5), 1.4)
    
    # 1) Détection avec Sobel
    print("\n1) Application de l'opérateur de Sobel...")
    sobel_edges, sobel_mag, sobel_orient = sobel_edge_detection(im_smooth, sb=30, sh=70)
    
    # 2) Détection avec Roberts
    print("2) Application de l'opérateur de Roberts...")
    roberts_edges, roberts_mag, roberts_orient = robert_edge_detection(im_smooth, sb=30, sh=70)
    
    # 3) Détection avec Prewitt
    print("3) Application de l'opérateur de Prewitt...")
    prewitt_edges, prewitt_mag, prewitt_orient = prewitt_edge_detection(im_smooth, sb=30, sh=70)
    
    # 4) Détection avec Canny (OpenCV)
    print("4) Application de l'algorithme de Canny (OpenCV)...")
    canny_edges = cv2.Canny(im_smooth, 30, 70)
    
    # Affichage des résultats
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Comparaison des détecteurs de contours', fontsize=16)
    
    # Ligne 1: Image originale et gradients
    axes[0, 0].imshow(im, cmap='gray')
    axes[0, 0].set_title('Image originale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sobel_mag, cmap='gray')
    axes[0, 1].set_title('Sobel - Amplitude du gradient')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sobel_orient, cmap='hsv')
    axes[0, 2].set_title('Sobel - Orientation du gradient')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(im_smooth, cmap='gray')
    axes[0, 3].set_title('Image lissée (Gaussien)')
    axes[0, 3].axis('off')
    
    # Ligne 2: Résultats des détecteurs personnalisés
    axes[1, 0].imshow(sobel_edges, cmap='gray')
    axes[1, 0].set_title('Sobel (personnalisé)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(roberts_edges, cmap='gray')
    axes[1, 1].set_title('Roberts (personnalisé)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(prewitt_edges, cmap='gray')
    axes[1, 2].set_title('Prewitt (personnalisé)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(canny_edges, cmap='gray')
    axes[1, 3].set_title('Canny (OpenCV)')
    axes[1, 3].axis('off')
    
    # Ligne 3: Comparaison avec différents seuils Canny
    canny_low = cv2.Canny(im_smooth, 20, 50)
    canny_high = cv2.Canny(im_smooth, 50, 100)
    canny_very_high = cv2.Canny(im_smooth, 100, 200)
    
    axes[2, 0].imshow(canny_low, cmap='gray')
    axes[2, 0].set_title('Canny (20, 50)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(canny_edges, cmap='gray')
    axes[2, 1].set_title('Canny (30, 70)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(canny_high, cmap='gray')
    axes[2, 2].set_title('Canny (50, 100)')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(canny_very_high, cmap='gray')
    axes[2, 3].set_title('Canny (100, 200)')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('CV_TP3/edge_detection_comparison.png', dpi=150, bbox_inches='tight')
    print("\nRésultats sauvegardés dans 'CV_TP3/edge_detection_comparison.png'")
    plt.show()
    
    # Analyse et remarques
    print("\n" + "="*80)
    print("ANALYSE DES RÉSULTATS:")
    print("="*80)
    print("\n1. Impact du bruit:")
    print("   - Les opérateurs de Roberts, Prewitt et Sobel sont sensibles au bruit")
    print("   - Un lissage gaussien en prétraitement améliore significativement les résultats")
    print("   - Canny intègre déjà un lissage gaussien, ce qui le rend plus robuste")
    
    print("\n2. Qualité des contours:")
    print("   - Roberts: Masques 2x2, détection rapide mais sensible au bruit")
    print("   - Prewitt: Masques 3x3, lissage dans une direction, moins sensible au bruit")
    print("   - Sobel: Masques 3x3 avec pondération, meilleur compromis bruit/précision")
    print("   - Canny: Optimal, contours fins et bien connectés grâce à l'hystérésis")
    
    print("\n3. Impact des seuils (Canny):")
    print("   - Seuils bas (20, 50): Plus de détails mais plus de bruit")
    print("   - Seuils moyens (30, 70): Bon équilibre")
    print("   - Seuils hauts (50, 100): Contours principaux uniquement")
    print("   - Seuils très hauts (100, 200): Contours les plus marqués seulement")
    
    print("\n4. Recommandations:")
    print("   - Toujours appliquer un lissage gaussien en prétraitement")
    print("   - Ajuster les seuils en fonction du niveau de bruit de l'image")
    print("   - Canny est généralement le meilleur choix pour la plupart des applications")
    print("="*80)


if __name__ == "__main__":
    # Test avec une image (à adapter selon les images disponibles)
    print("TP3 - Exercice 2: Segmentation d'image en contours")
    print("="*80)
    
    # Essayer différentes images
    test_images = [
        "Images/cameraman.bmp",
        "Images/lena.jpg",
        "Images/lena.png"
    ]
    
    image_found = False
    for img_path in test_images:
        try:
            test_edge_detectors(img_path)
            image_found = True
            break
        except:
            continue
    
    if not image_found:
        print("\nAucune image de test trouvée.")
        print("Veuillez placer une image (par exemple 'camera-man.jpg') dans le dossier 'Images/'")
        print("\nVous pouvez tester avec votre propre image en appelant:")
        print("  test_edge_detectors('chemin/vers/votre/image.jpg')")
