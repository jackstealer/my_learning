import cv2
import numpy as np

mask = cv2.imread("C:\Users\ATUL\Desktop\AGRICULT\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation\cc0000012.png", cv2.IMREAD_GRAYSCALE)
print(np.unique(mask))