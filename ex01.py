# ex01.py
import cv2
import numpy as np

image = cv2.imread('Images/Image_01.jpg')  # 读入图片
image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
cv2.imshow("Image", image2)  # 显示灰度图
# cv2.threshold(image2, 188, 255, 0, image2)  # cv2中的二值化函数


def my_threshold(img, thresholdValue):  # 定义二值化函数
    imgInFo = img.shape
    x = imgInFo[0]
    y = imgInFo[1]
    new_img = np.zeros(img.shape)
    for i in range(x):
        for j in range(y):
            if img[i, j] >= thresholdValue:  # 灰度值大于阈值的调整为255
                new_img[i, j] = 255
            else:  # 灰度值小于阈值的调整为0
                new_img[i, j] = 0
    return new_img


image3 = my_threshold(image2, 188)

cv2.imshow("Result", image3)  # 显示二值化的结果

cv2.waitKey(0)
cv2.imwrite('result_01_ori.jpg', image2)    # 保存当前灰度值处理过后的文件
cv2.imwrite('result_01_thr.jpg', image3)    # 保存当前灰度值处理过后的文件
