# ex02.py
import cv2

image1 = cv2.imread('Images/Image_02.jpg')  # 读入图片
image2 = cv2.imread('Images/Image_03.jpg')  # 读入图片
grey1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
grey2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)  # 转化为灰度图

cv2.imshow("Image1", image1)  # 图片显示
cv2.imshow("Image2", image2)  # 图片显示

image_add = cv2.add(image1, image2)  # 两个图片相加
grey_add = cv2.add(grey1, grey2)  # 两个灰度图相加
image_sub = cv2.subtract(image1, image2)  # 两个图片相减
grey_sub = cv2.subtract(grey1, grey2)  # 两个灰度图相减
image_addnum = cv2.add(image1, (50, 50, 50, 50))  # 加数字
image_subnum = cv2.subtract(image1, (50, 50, 50, 50))  # 减数字

cv2.imshow("AddResult", image_add)  # 图片显示
cv2.imshow("SubResult", image_sub)  # 图片显示
cv2.imshow("AddGreyResult", grey_add)  # 图片显示
cv2.imshow("SubGreyResult", grey_sub)  # 图片显示
cv2.imshow("AddNumResult", image_addnum)  # 图片显示
cv2.imshow("SubNumResult", image_subnum)  # 图片显示

cv2.waitKey(0)
cv2.imwrite('result_02_grey_01.jpg', grey1)    # 保存灰度图
cv2.imwrite('result_02_grey_02.jpg', grey2)    # 保存灰度图
cv2.imwrite('result_02_add.jpg', image_add)    # 保存当前相加处理过后的文件
cv2.imwrite('result_02_sub.jpg', image_sub)    # 保存当前相减处理过后的文件
cv2.imwrite('result_02_greyadd.jpg', grey_add)    # 保存当前灰度图相加处理过后的文件
cv2.imwrite('result_02_greysub.jpg', grey_sub)    # 保存当前灰度图相减处理过后的文件
cv2.imwrite('result_02_addnum.jpg', image_addnum)    # 保存当前加数字处理过后的文件
cv2.imwrite('result_02_subnum.jpg', image_subnum)    # 保存当前减数字处理过后的文件
