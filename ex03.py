# ex03.py
import cv2
import numpy as np
from PIL import Image

image = 'Images/Image_04.jpg'  # 读入图片
OriginalPic = np.array(Image.open(image).convert('L'), dtype=np.uint8)
img = np.zeros((OriginalPic.shape[0]+2, OriginalPic.shape[1]+2), np.uint8)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        img[i][j] = OriginalPic[i-1][j-1]  # 除去图像边缘

LaplacePic = np.zeros(
    (OriginalPic.shape[0], OriginalPic.shape[1]), dtype=np.uint8)
# kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]  # Laplace算子
# 扩展Laplace算子 加入对角线邻域使得对各方向边缘均有较强响应
kernel = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]

for i in range(0, LaplacePic.shape[0]):
    for j in range(0, LaplacePic.shape[1]):
        LaplacePic[i][j] = abs(
            np.sum(np.multiply(kernel, img[i:i+3, j:j+3])))  # 加权求和

cv2.imshow("Original", OriginalPic)  # 显示原图
cv2.imshow("Laplace", LaplacePic)  # 显示结果
cv2.waitKey(0)
cv2.imwrite('result_03_ori.jpg', OriginalPic)    # 保存OriginalPic 即原图像的灰度图
cv2.imwrite('result_03_lap.jpg', LaplacePic)    # 保存LaplacePic
