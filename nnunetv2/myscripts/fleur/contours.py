import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Definieer het pad naar de directory met afbeeldingen
directory_path = r'C:\Users\Test\Desktop\Fleur\Boundary_EdgeDetection'

# Loop door alle bestanden in de directory
for filename in os.listdir(directory_path):
    if filename.endswith('.png'):
        image_path = os.path.join(directory_path, filename)  # Volledig pad naar de afbeelding
        image = cv2.imread(image_path)

        # Controleer of de afbeelding is geladen
        if image is None:
            print(f"Afbeelding {filename} niet gevonden. Controleer het pad.")
            continue

        # Converteer de afbeelding naar HSV-kleurmodel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Definieer het bereik voor de rode kleur
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Maak een masker voor de rode kleur
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Vind de contouren
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Bepaal de pieken en dalen
        if contours:
            # Neem de grootste contour aan als de relevante
            contour = max(contours, key=cv2.contourArea)

            # Verkrijg de y-coördinaten van de contour
            contour = contour[:, 0, :]  # Houd alleen de x, y-coördinaten
            y_coords = contour[:, 1]

            # Tel het aantal pieken en dalen
            peaks = (np.diff(np.sign(np.diff(y_coords))) == -2).nonzero()[0] + 1  # Piek
            valleys = (np.diff(np.sign(np.diff(y_coords))) == 2).nonzero()[0] + 1  # Dal

            print(f"Afbeelding: {filename} - Aantal pieken: {len(peaks)}, Aantal dalen: {len(valleys)}")

            # Visualiseer de originele afbeelding en de contouren
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Originele Afbeelding')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.plot(contour[:, 0], contour[:, 1], color='blue', label='Contour')
            plt.scatter(contour[peaks, 0], contour[peaks, 1], color='red', label='Pieken', zorder=5)
            plt.scatter(contour[valleys, 0], contour[valleys, 1], color='green', label='Dalen', zorder=5)
            plt.title('Gevonden Contouren')
            plt.axis('off')
            plt.legend()

            plt.show()
        else:
            print(f"Geen contouren gevonden in afbeelding: {filename}.")
