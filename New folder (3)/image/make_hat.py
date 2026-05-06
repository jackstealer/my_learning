import cv2
import numpy as np
import os

src = r'C:\Users\ATUL\.gemini\antigravity\brain\dce1438e-59d3-4312-afcd-e3b7c7bf9306\hat_overlay_1777627849698.png'
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hat.png')

img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
print('Input shape:', img.shape)

# Convert to BGRA if needed
if img.shape[2] == 3:
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
elif img.shape[2] == 4:
    img_bgra = img.copy()

# Remove white/near-white background -> make transparent
white_mask = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
img_bgra[white_mask, 3] = 0  # fully transparent

cv2.imwrite(dst, img_bgra)
print('Saved hat.png with transparency to:', dst)
