# ex05.py
import cv2
import numpy as np


def func_o(img1, img):
    B, G, R = cv2.split(img1)
    B_mean = np.mean(B)
    G_mean = np.mean(G)
    R_mean = np.mean(R)
    O_B = np.array(B, copy=True)
    O_G = np.array(G, copy=True)
    O_R = np.array(R, copy=True)
    O_O = np.array(B, copy=True)
    # 分别计算三种基色的方差
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            O_B[i, j] = (img1[i, j, 0] - B_mean)**2
            O_G[i, j] = (img1[i, j, 1] - G_mean)**2
            O_R[i, j] = (img1[i, j, 2] - R_mean)**2
            O_O[i, j] = pow((O_B[i, j]**2 + O_G[i, j]**2 + O_R[i, j]**2), 0.5)
    O_avg = np.sum(O_O) / (img1.shape[0]*img1.shape[1])  # 计算样图的欧几里得距离的均值 即为中心点
    # O_std = np.std(O_O)  # 计算样图欧几里得距离的标准差 作为阈值
    gray_img = np.ones([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            D = pow(((img[i, j, 0]-B_mean)**2 + (img[i, j, 1]-G_mean) **
                    2 + (img[i, j, 2]-R_mean)**2), 0.5)  # 计算当前区域的欧几里得距离 与阈值作比较
            if(D <= O_avg):
                gray_img[i, j] = 255  # 阈值范围内
            else:
                gray_img[i, j] = 0  # 阈值范围外
    color_img = np.array(img, copy=True)
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            if (gray_img[i, j] == 0):  # 删除阈值范围外的颜色
                color_img[i, j, 0] = 0
                color_img[i, j, 1] = 0
                color_img[i, j, 2] = 0
    return color_img  # 返回提取相应颜色后的图片


def func_a(img1, img):
    B, G, R = cv2.split(img1)
    B_mean = np.mean(B)
    G_mean = np.mean(G)
    R_mean = np.mean(R)
    A_B = np.array(B, copy=True)
    A_G = np.array(G, copy=True)
    A_R = np.array(R, copy=True)
    A_A = np.array(R, copy=True)
    # 分别计算三种基色的方差
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            A_B[i, j] = abs(img1[i, j, 0] - B_mean)
            A_G[i, j] = abs(img1[i, j, 1] - G_mean)
            A_R[i, j] = abs(img1[i, j, 2] - R_mean)
            A_A[i, j] = int(A_B[i, j]) + int(A_G[i, j]) + int(A_R[i, j])
    # 计算样图的绝对值距离的均值 即为中心点
    A_avg = np.sum(A_A) / (img1.shape[0] * img1.shape[1])
    A_std = np.std(A_A)  # 计算样图绝对距离的标准差 作为阈值
    gray_img = np.ones([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            D = abs(img[i, j, 0]-B_mean)+abs(img[i, j, 1]-G_mean) + \
                abs(img[i, j, 2]-R_mean)  # 计算当前区域的绝对值距离 与阈值作比较
            if(abs(D - A_avg) <= A_std):
                gray_img[i, j] = 255  # 阈值范围内
            else:
                gray_img[i, j] = 0  # 阈值范围外
    color_img = np.array(img, copy=True)
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            if (gray_img[i, j] == 0):  # 删除阈值范围外的颜色
                color_img[i, j, 0] = 0
                color_img[i, j, 1] = 0
                color_img[i, j, 2] = 0
    return color_img  # 返回提取相应颜色后的图片


if __name__ == '__main__':
    strawberry = cv2.imread('Images/strawberry.jpg')  # 读入截取的草莓图片
    image = cv2.imread('Images/Image_05.jpg')  # 读入原图
    cv2.imshow("Image", image)  # 显示原图
    result_o = func_o(strawberry, image)  # 调用定义的欧氏距离算法函数
    result_a = func_a(strawberry, image)  # 调用定义的绝对距离算法函数
    cv2.imshow("Result_o", result_o)  # 显示结果
    cv2.imshow("Result_a", result_a)  # 显示结果

    cv2.waitKey(0)
    cv2.imwrite('result_05_o.jpg', result_o)  # 保存欧氏距离算法结果
    cv2.imwrite('result_05_a.jpg', result_a)  # 保存绝对距离算法结果
