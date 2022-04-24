# ex04.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('Images/Image_04.jpg')  # 读入图片
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
edges = cv2.Canny(image, 80, 200)  # Canny算法
# 阈值下界minVal 阈值上界maxVal
# 图像中的像素点如果大于阈值上界则认为必然是边界 即强边界
# 小于阈值下界则认为必然不是边界
# 两者之间的则认为是候选项 即弱边界
cv2.imshow("Image", image)  # 图片显示
cv2.imshow("Result", edges)  # 图片显示

cv2.waitKey(0)
cv2.imwrite('result_04.jpg', edges)    # 保存当前Canny算法处理过后的文件
