---
Profile : 2019级计算机科学与技术1班 高润 1925101007
Date : 2022 / 04 / 23
---

[TOC]

# 图形图像编程实践  课程报告

### 实验环境

- Windows 11
- Visual Studio Code
- Python 3.10.0
- OpenCV + NumPy

## EX01  图像的二值化

### 问题描述

编程实现图像的二值化，分析不同的阈值对二值化图像的影响

T=f(x,y)，其中x,y空间坐标，T代表灰度

### 算法设计

1. 可以直接调用cv2中的二值化函数。自己设定阈值参数，灰度值大于阈值的将灰度值设为255，即白色；反之则设为0，即黑色。根据结果调整阈值到一个合适的值，使结果更美观

```python
cv2.threshold(image2, 188, 255, 0, image2)  # cv2中的二值化函数
```

2. 自己定义二值化函数，计算思想同上。源代码如下：

```python
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
```

### 结果分析

#### 原图

<img src="https://s2.loli.net/2022/04/23/YhM95Q4xvkHzJAg.jpg" alt="Image_01" style="zoom: 33%;" />

### 灰度图

<img src="https://s2.loli.net/2022/04/23/oBEAGMbuNYqnFct.jpg" alt="result_01_ori" style="zoom:33%;" />

### 二值化结果图

<img src="https://s2.loli.net/2022/04/23/MSbHn4hyvd7J1Gr.jpg" alt="result_01_thr" style="zoom:33%;" />

- 经过不断调整，阈值设为188时，能够比较好的显示出原图的内容

## EX02  图像的加减

### 问题描述

编程实现图像的基本运算：两幅图像相加和相减，并分析这两种运算的作用

1. 灰度/RGB图像

2. 相加（数据）

3. 相减（数据）

### 算法设计

- 相加：两张图片每个像素点的RGB值相加，得到的和对255取模为结果图片对应像素点的RGB值
- 图像的加法运算是将一幅图像的内容叠加在另一幅图像上，或者给图像的每一个像素加一个常数来改变图像的亮度

- 相减：两张图片每个像素点的RGB值相减，得到的差的绝对值为结果图片对应像素点的RGB值
- 图像相减可以检测出两幅图像的差异信息，因此这项技术在工业、医学、气象以及军事等领域中都有广泛的应用
- 在cv2中含有图像加减法的函数可以直接调用

```python
image1 = cv2.imread('Images/Image_02.jpg')  # 读入图片
image2 = cv2.imread('Images/Image_03.jpg')  # 读入图片
grey1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
grey2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)  # 转化为灰度图

image_add = cv2.add(image1, image2)  # 两个图片相加
grey_add = cv2.add(grey1, grey2)  # 两个灰度图相加
image_sub = cv2.subtract(image1, image2)  # 两个图片相减
grey_sub = cv2.subtract(grey1, grey2)  # 两个灰度图相减
image_addnum = cv2.add(image1, (50, 50, 50, 50))  # 加数字
image_subnum = cv2.subtract(image1, (50, 50, 50, 50))  # 减数字

cv2.imwrite('result_02_grey_01.jpg', grey1)    # 保存灰度图
cv2.imwrite('result_02_grey_02.jpg', grey2)    # 保存灰度图
cv2.imwrite('result_02_add.jpg', image_add)    # 保存当前相加处理过后的文件
cv2.imwrite('result_02_sub.jpg', image_sub)    # 保存当前相减处理过后的文件
cv2.imwrite('result_02_greyadd.jpg', grey_add)    # 保存当前灰度图相加处理过后的文件
cv2.imwrite('result_02_greysub.jpg', grey_sub)    # 保存当前灰度图相减处理过后的文件
cv2.imwrite('result_02_addnum.jpg', image_addnum)    # 保存当前加数字处理过后的文件
cv2.imwrite('result_02_subnum.jpg', image_subnum)    # 保存当前减数字处理过后的文件
```

### 结果分析

#### 原图

<img src="https://s2.loli.net/2022/04/23/AbfdveM8yt4LaV2.jpg" alt="Image_02" style="zoom: 25%;" />

<img src="E:/work/ImgWork/Images/image_03.jpg" alt="image_03" style="zoom:25%;" />

#### 灰度图

<img src="https://s2.loli.net/2022/04/23/YsvDjZI2dXlMSry.jpg" alt="result_02_grey_01" style="zoom:25%;" />

<img src="https://s2.loli.net/2022/04/23/T12coIUEfmaHNVr.jpg" alt="result_02_grey_02" style="zoom:25%;" />

#### 图片加减结果图

##### RPG图像加减

<img src="https://s2.loli.net/2022/04/23/bSn8H2fUIiJat75.jpg" alt="result_02_add" style="zoom:25%;" />

<img src="https://s2.loli.net/2022/04/23/RrelQ7nbV3ZFqXt.jpg" alt="result_02_sub" style="zoom:25%;" />

##### 灰度图加减

<img src="https://s2.loli.net/2022/04/23/gXCEubrW7GieBdh.jpg" alt="result_02_greyadd" style="zoom:25%;" />

<img src="https://s2.loli.net/2022/04/23/CAgqmW7NjRK6csQ.jpg" alt="result_02_greysub" style="zoom:25%;" />

#### 图像与数据相加减

<img src="https://s2.loli.net/2022/04/23/wn9LxdNVl21tIDF.jpg" alt="result_02_addnum" style="zoom:25%;" />

<img src="https://s2.loli.net/2022/04/23/Y9ebpUXFJZs2QaS.jpg" alt="result_02_subnum" style="zoom:25%;" />

- 不难看出，图像与数据相加就是提高亮度，与数据相减就是降低亮度
- 而图像与图像的相减，无论是RGB图像还是灰度图像，都是两张图图形的合成，若两张图像中存在完全相同的部分，相减则会使该部分消失，因此在实际应用中可用于图像的降噪处理

## EX03  图像拉普拉斯锐化

### 问题描述

编程实现图像拉普拉斯锐化

拉普拉斯模板如何实现

### 算法设计

- 二维函数f(x,y)的二阶微分定义为

$$
\Delta^2 f(x,y) = \delta^2 f / \delta x^2 + \delta^2 f / \delta y^2
$$

- 用差分来代替微分

$$
\delta^2 f / \delta x^2 = [ f(i + 1, j) - f(i, j) ] - [ f(i, j) - f(i - 1, j) ] = f(i + 1, j) + f(i - 1, j) - 2f(i, j)
$$

$$
\delta^2 f / \delta y^2 = [ f(i, j + 1) - f(i, j) ] - [ f(i, j) - f(i, j - 1) ] = f(i, j + 1) + f(i, j - 1) - 2f(i, j)
$$

- 两式相加就得到了用于图像锐化的拉普拉斯算子

$$
\Delta^2 f(x,y) = f(i + 1, j) + f(i - 1, j) + f(i, j + 1) + f(i, j - 1) - 4f(i, j)
$$

- 对应的滤波模板为

$$
kernel =  \begin{bmatrix}
   0 & 1 & 0 \\
   1 & -4 & 1 \\
   0 & 1 & 0
  \end{bmatrix}
$$

- 拉普拉斯算子旋转90度等于自身，也就说明它对接近水平和接近竖直方向的边缘都有很好的加强。更进一步，我们构造对于45度旋转各向同性的滤波器如下

$$
kernel' =  \begin{bmatrix}
   1 & 1 & 1 \\
   1 & -8 & 1 \\
   1 & 1 & 1
  \end{bmatrix}
$$

- 代码实现如下

```python
image = 'Images/Image_04.jpg'  # 读入图片
OriginalPic = np.array(Image.open(image).convert('L'), dtype=np.uint8)
img = np.zeros((OriginalPic.shape[0]+2, OriginalPic.shape[1]+2), np.uint8)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        img[i][j] = OriginalPic[i-1][j-1]  # 除去图像边缘

LaplacePic = np.zeros(
    (OriginalPic.shape[0], OriginalPic.shape[1]), dtype=np.uint8)
kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]  # Laplace算子
# kernel = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]  # 扩展Laplace算子 加入对角线邻域使得对各方向边缘均有较强响应

for i in range(0, LaplacePic.shape[0]):
    for j in range(0, LaplacePic.shape[1]):
        LaplacePic[i][j] = abs(np.sum(np.multiply(kernel, img[i:i+3, j:j+3])))  # 加权求和
```

### 结果分析

#### 原图

<img src="https://s2.loli.net/2022/04/23/lSrUCgXt2VHeB1j.jpg" alt="Image_04" style="zoom: 50%;" />

#### 灰度图

<img src="https://s2.loli.net/2022/04/23/recultKAjzGFY67.jpg" alt="result_03_ori" style="zoom:50%;" />

#### 拉普拉斯锐化结果图

<img src="https://s2.loli.net/2022/04/23/lpxBU4oAeDW6Hcm.jpg" alt="result_03_lap" style="zoom: 50%;" />

#### 扩展拉普拉斯算子结果图

<img src="https://s2.loli.net/2022/04/23/ezur57iap9Z8HtR.jpg" alt="result_03_lap" style="zoom:50%;" />

- 拉普拉斯锐化后的灰度图仅剩下较为明显的边界，但不够清晰
- 在扩展了拉普拉斯算子后，得到的结果明显优于扩展之前的，拥有较为清晰的边界

## EX04  Canny图像边缘检测

### 问题描述

编程实现Canny图像边缘检测方法，对不同参数进行验证。

1. 对灰度图像，--实现Canny的计算过程，（MATLAB, Python, OpenCV）

2. Canny里面有大小2个阈值，分析这两个值对结果的影响。

### 算法设计

- 阈值下界minVal 阈值上界maxVal

- 图像中的像素点如果大于阈值上界则认为必然是边界 即强边界

- 小于阈值下界则认为必然不是边界

- 两者之间的则认为是候选项 即弱边界
- 调用cv2中的Canny函数完成边界检测

```
image = cv2.imread('Images/Image_04.jpg')  # 读入图片
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
edges = cv2.Canny(image, 80, 200)  # Canny算法
```

### 结果分析

#### 原图

<img src="https://s2.loli.net/2022/04/23/lSrUCgXt2VHeB1j.jpg" alt="Image_04" style="zoom: 50%;" />

#### 灰度图

<img src="https://s2.loli.net/2022/04/23/recultKAjzGFY67.jpg" alt="result_03_ori" style="zoom:50%;" />

#### 阈值80，200

<img src="https://s2.loli.net/2022/04/23/VvchoDy7wpI2Agn.jpg" alt="result_04" style="zoom:50%;" />

#### 阈值120，220

<img src="https://s2.loli.net/2022/04/23/3zn5l9Eek6imbWg.jpg" alt="result_04" style="zoom:50%;" />

#### 阈值230，255

<img src="https://s2.loli.net/2022/04/23/qTeEXujlzraAoch.jpg" alt="result_04" style="zoom:50%;" />

- 经过多次调整阈值，根据结果不难看出，阈值设定越高，对边界的要求就越高，得到的结果图中边界就越少，在实际应用当中也需要进行多次测试找到一个相对合适的阈值以取得最佳效果

## EX05  彩色图像的切割

### 问题描述

编程实现彩色图像的切割。设计一个通用的方法，基于RGB空间，从下列的图像中，分割出指定的目标（如草莓）。

1. 特征：RGB向量（3维）；

2. 计算特征的距离—定义草莓的空间（球：欧式距离，立方体：绝对值距离），分析这两种不同的距离的差异。

欧式距离：中心点，半径阈值R

绝对值距离：中心点，距离阈值R

中心点定义：[255,0,0] ； 随机截取一个草莓区域，这个区域的RGB向量的均值作为中心点，阈值，可以用方差表示。

<img src="https://s2.loli.net/2022/04/23/jnZfYDRyXQEtT37.jpg" alt="Image_05" style="zoom: 33%;" />

### 算法设计

- 比较每个像素点与所需颜色空间的相似程度，在阈值范围内的则保留，不在的则赋黑色，实现提取目标物体的需求
- - 欧氏距离

指在m维空间中**两个点之间的真实距离**，或者**向量的自然长度**（即该点到原点的距离）

比如：在二维和三维空间中的欧氏距离就是两点之间的实际距离

由于RGB模型为三维空间，因此本实验中求欧氏距离的公式如下
$$
d_1(point_1, point_2) = \sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (G_1 - G_2)^2}
$$
定义算法函数如下

```python
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
    O_std = np.std(O_O)  # 计算样图欧几里得距离的标准差 作为阈值
    gray_img = np.ones([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            D = pow(((img[i, j, 0]-B_mean)**2 + (img[i, j, 1]-G_mean) **
                    2 + (img[i, j, 2]-R_mean)**2), 0.5)  # 计算当前区域的欧几里得距离 与阈值作比较
            if((D - O_avg)**2 <= O_std):
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
```

- - 绝对值距离

指在m维空间中两点各方向坐标的差的绝对值之和

由于RGB模型为三维空间，因此本实验中求绝对值距离的公式如下
$$
d_2(point_1, point_2) = \vert R_1 - R_2 \vert + \vert G_1 - G_2 \vert + \vert B_1 - B_2 \vert
$$
定义算法函数如下

```python
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
            A_A[i, j] = A_B[i, j] + A_G[i, j] + A_R[i, j]
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
```

- 主函数

```python
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
```

### 结果分析

#### 原图

<img src="https://s2.loli.net/2022/04/23/jnZfYDRyXQEtT37.jpg" alt="Image_05" style="zoom: 33%;" />

#### 截取的样图

![Strawberry](https://s2.loli.net/2022/04/28/SdjABZq7wtHsYbM.jpg)

![Leaf](https://s2.loli.net/2022/04/28/GAM4Td6zQDeb9Rr.jpg)

#### 欧氏距离算法提取结果

<img src="https://s2.loli.net/2022/04/23/OkSBUvsAfEbXZTl.jpg" alt="result_05_o" style="zoom: 33%;" />

#### 绝对距离算法提取结果

<img src="https://s2.loli.net/2022/04/23/a5flnNKIGU6bJ9e.jpg" alt="result_05_a" style="zoom:33%;" />

- 两种算法的结果都有一个共同点：右下角的咖啡也被识别为草莓而提取出来，原因是颜色较为相近，可能也是使用RGB模型进行提取的劣势所在

- 不考虑其余物品颜色干扰的情况下，欧氏距离算法提取后的效果相对优于绝对距离算法

- 自然环境下获取的图像容易受自然光照、遮挡和阴影等情况的影响，即对亮度比较敏感，而RGB颜色空间的三个分量都与亮度密切相关，即只要亮度改变，三个分量都会随之相应地改变，而没有一种更直观的方式来表达在HSV颜色空间下，比 BGR 更容易跟踪某种颜色的物体，常用于分割指定颜色的物体，因此先把RGB转化为HSV，便于提取颜色的特征

- 考虑到RGB模型下提取草莓的效果不太好，经过查阅资料，使用HSV模型提取的算法如下：

  分别提取草莓的红色果肉和绿色叶子，最后合成为一张图片，源代码如下：
  
```python
# 取色器
# http://www.jiniannet.com/Page/allcolor
# 将获取的四个点的rgb值填入下方的rgb，即可得到四个点的hsv上下限
import cv2
import numpy as np

rgb = '#ACB064,#AAA95F,#7B7129,#6B6835'

rgb = rgb.split(',')

# 转换为BGR格式，并将16进制转换为10进制
bgr = [[int(r[5:7], 16), int(r[3:5], 16), int(r[1:3], 16)] for r in rgb]

# 转换为HSV格式
hsv = [list(cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2HSV)[0][0])
       for b in bgr]

hsv = np.array(hsv)
print('H:', min(hsv[:, 0]), max(hsv[:, 0]))
print('S:', min(hsv[:, 1]), max(hsv[:, 1]))
print('V:', min(hsv[:, 2]), max(hsv[:, 2]))

```

```python
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
```

#### 在HSV空间提取草莓结果

<img src="https://s2.loli.net/2022/04/23/m3PaxSBfdXj9DKl.jpg" alt="result_05_hsv" style="zoom:33%;" />

- 不难看出，在HSV空间下提取的草莓没有受其他物品的干扰，草莓有残缺是阈值设定不够精确的结果，还需继续改进

## 完整源代码

### EX01

```python
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

```

### EX02

```python
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

```

### EX03

```python
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

```

### EX04

```python
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

```

### EX05

```python
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
            A_A[i, j] = A_B[i, j] + A_G[i, j] + A_R[i, j]
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

```

```python
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

```



