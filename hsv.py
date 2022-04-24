# hsv.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('Images/Image_05.jpg')  # 读入图片
cv2.imshow("Image", image)  # 显示原图

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 色彩空间转换

# 红色和绿色的HSV的阈值
light_red = (0, 170, 150)
dark_red = (180, 230, 230)
light_green = (25, 90, 90)
dark_green = (35, 200, 190)

red_mask = cv2.inRange(hsv_img, light_red, dark_red)  # 提取草莓红色果肉
green_mask = cv2.inRange(hsv_img, light_green, dark_green)  # 提取草莓绿色叶子
mask = red_mask+green_mask
result = cv2.bitwise_and(image, image, mask=mask)  # 合为一张图片

cv2.imshow("Result", result)  # 显示提取后的结果
cv2.waitKey(0)
cv2.imwrite('result_05_hsv.jpg', result)  # 保存结果
