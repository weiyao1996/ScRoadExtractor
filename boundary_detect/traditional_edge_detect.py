import cv2
import os

img_path = '/data/train/sat/'
edge_path = '/data/train/rough_edge_canny/'
os.makedirs(edge_path, exist_ok=True)

img_name = os.listdir(img_path)
for name in img_name:
    img = cv2.imread(img_path + name, 0)

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    canny_edges = cv2.Canny(blur, 20, 200)

    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # Scale_absX = cv2.convertScaleAbs(x)
    # Scale_absY = cv2.convertScaleAbs(y)
    # sobel_edges = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

    cv2.imwrite(edge_path + name[:-7] + 'hed.png', canny_edges)
