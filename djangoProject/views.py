import random
from tkinter import Image
import random
from tkinter import Image
import dlib
import cv2
import cv2 as cv
import numpy as np
import cmake
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from imutils import face_utils
from matplotlib import pyplot as plt
from djangoProject import settings


# 主页面
def to_img_load(request):
    return render(request, 'upload.html')


# 人脸识别
def faceDetect(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        print(name)
        path = settings.MEDIA_ROOT + '/' + image.name
        with open(path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = np.float32(gray)

        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        face_detect = dlib.get_frontal_face_detector()
        rects = face_detect(gray, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.imwrite(path, gray)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)



# 绘制直方图
def histogram(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        print(name)
        path = settings.MEDIA_ROOT + '/' + image.name
        with open(path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(path)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.savefig(path)
        try:
            data = {'state': 'static/media/'+name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 灰度直方图
def greyHistogram(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + image.name
        print(img_path)
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img_PIL = Image.open(img_path)
        img = np.array(img_PIL)
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        hist = cv.calcHist([img_gray], [0], None, [256], [0, 255])
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.savefig(img_path)
        try:
            data = {'state': 'static/media/'+name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 彩色直方图
def colorHistogram(request):
    if request.method == 'POST':
        color = ["r", "g", "b"]
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + image.name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        # 绘制彩色直方图，需要对每个通道进行遍历，并且找到最大值和最小值
        for index, c in enumerate(color):
            hist = cv2.calcHist([img], [index], None, [256], [0, 255])
            plt.plot(hist, color=c)
            plt.xlim([0, 255])
        plt.savefig(img_path)
        try:
            data = {'state': 'static/media/'+name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 分段线性处理
def piecewiseLinearProcessing(request):
    # 定义图像数据的路径
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path, 0)
        h, w = img.shape[:2]
        out = np.zeros(img.shape, np.uint8)
        for i in range(h):
            for j in range(w):
                pix = img[i][j]
                if pix < 50:
                    out[i][j] = 0.5 * pix
                elif pix < 150:
                    out[i][j] = 3.6 * pix - 310
                else:
                    out[i][j] = 0.238 * pix + 194
        out = np.around(out)
        out = out.astype(np.uint8)
        cv2.imwrite(img_path, out)
        try:
            data = {'state': 'static/media/'+name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 放大图像至原来的两倍，使用双线性插值法????
def enlarge(request):
    # 定义图像数据的路径
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        l, w, h = img.shape
        img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(img_path, img)
        try:
            data = {'state': 'static/media/'+name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 图片平移 构建移动矩阵,x轴左移 10 个像素，y轴下移 30 个
def move(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        M = np.float32([[1, 0, 30], [0, 1, 60]])
        img = cv2.warpAffine(img, M, (width, height))
        cv2.imwrite(img_path, img)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 构建矩阵，旋转中心坐标为处理后图片长宽的一半，旋转角度为45度，缩放因子为1
def spin(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        rows, cols, depth = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        dst = cv2.warpAffine(img, M, (width, height))
        cv2.imwrite(img_path, dst)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 水平翻转
def horizontalFlip(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        horizontal = cv2.flip(img, 1, dst=None)
        cv2.imwrite(img_path, horizontal)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 垂直翻转
def verticalFlip(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        vertical = cv2.flip(img, 0, dst=None)
        cv2.imwrite(img_path, vertical)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 对角线翻转
def crossFlip(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        cross = cv2.flip(img, -1, dst=None)
        cv2.imwrite(img_path, cross)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 仿射变换
def affineTransformation(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        rows, cols = img.shape[: 2]
        # 设置图像仿射变化矩阵
        post1 = np.float32([[50, 50], [200, 50], [50, 200]])
        post2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(post1, post2)
        # 图像仿射变换，及保存
        result = cv2.warpAffine(img, M, (rows, cols))
        cv2.imwrite(img_path, result)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 图片增强
def enhance(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        # 1. 灰度模式读取图像，图像名为CRH
        CRH = cv2.imread(img_path, 0)

        # 2. 计算图像梯度。首先要对读取的图像进行数据变换，因为使用了
        # numpy对梯度进行数值计算，所以要使用
        # CRH.astype('float')进行数据格式变换。
        row, col = CRH.shape
        CRH_f = np.copy(CRH)
        CRH_f = CRH_f.astype('float')
        gradient = np.zeros((row, col))
        for x in range(row - 1):
            for y in range(col - 1):
                gx = abs(CRH_f[x + 1, y] - CRH_f[x, y])
                gy = abs(CRH_f[x, y + 1] - CRH_f[x, y])
                gradient[x, y] = gx + gy
        # 3. 对图像进行增强，增强后的图像变量名为sharp
        sharp = CRH_f + gradient
        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        # 数据类型变换
        gradient = gradient.astype('uint8')
        sharp = sharp.astype('uint8')
        cv2.imwrite(img_path, sharp)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# robs算子图片增强
def robs(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        # 1. 灰度化处理图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. Roberts算子
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        # 3. 卷积操作
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        # 4. 数据格式转换
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        Roberts = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        cv2.imwrite(img_path, Roberts)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 使用 Sobel 算子提取边缘
def sob(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        # 1. 灰度化处理图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 求Sobel 算子
        sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        # 3. 数据格式转换
        absx = cv2.convertScaleAbs(sx)
        absy = cv2.convertScaleAbs(sy)
        # 4. 组合图像
        Sobel = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        cv2.imwrite(img_path, Sobel)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 使用 Laplacian 算子提取边缘
def lap(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        # 1. 灰度化处理图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 高斯滤波
        grayG = cv2.GaussianBlur(gray, (5, 5), 0)
        fltI = cv2.Laplacian(grayG, cv2.CV_16S, ksize=3)
        # 3. 拉普拉斯算法
        Laplacian = cv2.convertScaleAbs(fltI)
        cv2.imwrite(img_path, Laplacian)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 使用 LoG 算子提取边缘
def log(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.copyMakeBorder(gray, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
        image = cv2.GaussianBlur(gray, (3, 3), 0, 0)
        m1 = np.array([[0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0]])
        row = image.shape[0]
        col = image.shape[1]
        image1 = np.zeros(image.shape)
        for i in range(2, row - 2):
            for j in range(2, col - 2):
                image1[i, j] = np.sum(m1 * image[i - 2:i + 3, j - 2:j + 3, 1])
        image1 = cv2.convertScaleAbs(image1)
        cv2.imwrite(img_path, image1)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# Canny 边缘检测过程
def cny(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        src = cv2.imread(img_path)
        blur = cv2.GaussianBlur(src, (3, 3), 0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        gradx = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
        grady = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
        edge_output = cv2.Canny(gradx, grady, 50, 150)
        cv2.imwrite(img_path, edge_output)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 均值滤波器
def MeanFilter(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w = gray.shape[1]
        h = gray.shape[0]
        newing = np.array(gray)

        lbing = np.zeros((h + 2, w + 2), np.float32)
        tmping = np.zeros((h + 2, w + 2))
        myh = h + 2
        myw = w + 2
        tmping[1:myh - 1, 1:myw - 1] = newing[0:h, 0:w]

        a = 1 / 8.0
        kernel = a * np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        for y in range(1, myh - 1):
            for x in range(1, myw - 1):
                lbing[y, x] = np.sum(kernel * tmping[y - 1:y + 2, x - 1:x + 2])

        result = np.array(lbing[1:myh - 1, 1:myw - 1], np.uint8)
        cv2.imwrite(img_path, result)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 中值滤波器
def MedFilter(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        w = img.shape[1]
        h = img.shape[0]
        median = np.array(img)
        for y in range(1, w):
            for x in range(1, h):
                median[x, y] = np.median(img[x - 1:x + 2, y - 1:y + 2])
        cv2.imwrite(img_path, median)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# Hough线段变化
def HoughLineChange(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/2, 118)

        result = img.copy()
        for i_line in lines:
            for line in i_line:
                rho = line[0]
                theta = line[1]
                if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                    pt1 = (int(rho / np.cos(theta)), 0)
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    cv2.line(result, pt1, pt2, (0, 0, 255))
                else:
                    pt1 = (0, int(rho / np.sin(theta)))
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    cv2.line(result, pt1, pt2, (0, 0, 255), 1)


        minLineLength = 200
        maxLineGap = 15


        linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength, maxLineGap)

        result_P = img.copy()
        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(img_path, result_P)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 腐蚀
def erode(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用一个5x5的交叉型结构元（核心在几何中心）对二值图片src进行腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        erosion = cv2.erode(src, kernel)
        cv2.imwrite(img_path, erosion)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 膨胀
def dialate(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用一个5x5的交叉型结构元（核心在几何中心）对二值图片src进行膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        dilation = cv2.dilate(src, kernel)
        cv2.imwrite(img_path, dilation)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 开操作######
def openMorphing(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        #   定义十字形结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))

        im_op = cv2.morphologyEx(dst, cv2.MORPH_OPEN,kernel)
        cv2.imwrite(img_path, im_op)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 闭操作
def closeMorphing(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        #   定义十字形结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))

        im_cl = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(img_path, im_cl)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 雪花噪声
def sp_noise(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        prob = 0.2
        # 待输出的图片
        output = np.zeros(img.shape, np.uint8)

        # 遍历图像，获取叠加噪声后的图像
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        cv2.imwrite(img_path, output)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 添加高斯噪声,mean : 均值 , var : 方差
def gasuss_noise(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        image = np.array(img / 255, dtype=float)
        var = 0.001
        mean = 0
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        cv2.imwrite(img_path, out)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# 高通滤波
def highPassFilter(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = np.zeros(image.shape, np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if 200 < image[i][j]:
                    output[i][j] = image[i][j]
                else:
                    output[i][j] = 0
        cv2.imwrite(img_path, output)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


##########
##########
# 空域锐化
##########
#########



def sharpen(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        dst = cv2.filter2D(img, -1, kernel=kernel)
        cv2.imwrite(img_path, dst)
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


##########
##########
# 频域锐化
##########
#########

# 理想低通滤波
def IdealLowPassFiltering(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        f_shift = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 设置滤波半径
        D0 = 80
        # 初始化
        m = f_shift.shape[0]
        n = f_shift.shape[1]
        h1 = np.zeros((m, n))
        x0 = np.floor(m / 2)
        y0 = np.floor(n / 2)
        for i in range(m):
            for j in range(n):
                D = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                if D <= D0:
                    h1[i][j] = 1
        cv2.imwrite(img_path, frequency_filter(f_shift, h1))
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# Butterworth低通滤波器
def butterworth_low_filter(request):
    """
        生成一个Butterworth低通滤波器（并返回）
    """
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        D0 = 20
        rank = 2
        h, w = img.shape[:2]
        filter_img = np.zeros((h, w))
        u = np.fix(h / 2)
        v = np.fix(w / 2)
        for i in range(h):
            for j in range(w):
                d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
                filter_img[i, j] = 1 / (1 + 0.414 * (d / D0) ** (2 * rank))
        cv2.imwrite(img_path, frequency_filter(img, filter_img))
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


##########
##########
# 频域平滑
##########
#########


# 理想高通滤波
def IdealHighPassFiltering(request):
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        f_shift = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 设置滤波半径
        D0 = 80
        # 初始化
        m = f_shift.shape[0]
        n = f_shift.shape[1]
        h1 = np.zeros((m, n))
        x0 = np.floor(m / 2)
        y0 = np.floor(n / 2)
        for i in range(m):
            for j in range(n):
                D = np.sqrt((i - x0) ** 2 + (j - y0) ** 2)
                if D >= D0:
                    h1[i][j] = 1
        cv2.imwrite(img_path, frequency_filter(f_shift, h1))
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


# Butterworth高通滤波器
def butterworth_high_filter(request):
    """
        生成一个Butterworth高通滤波器（并返回）
    """
    if request.method == 'POST':
        image = request.FILES.get('picture')
        name = image.name
        img_path = settings.MEDIA_ROOT + '/' + name
        with open(img_path, 'wb') as pic:
            for c in image.chunks():
                pic.write(c)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        D0 = 40
        rank = 2
        h, w = img.shape[:2]
        filter_img = np.zeros((h, w))
        u = np.fix(h / 2)
        v = np.fix(w / 2)
        for i in range(h):
            for j in range(w):
                d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
                filter_img[i, j] = 1 / (1 + (D0 / d) ** (2 * rank))
        cv2.imwrite(img_path, frequency_filter(img, filter_img))
        try:
            data = {'state': 'static/media/' + name}
        except:
            data = {'state': 0}
        return JsonResponse(data)


def frequency_filter(image, filter):
    """
    :param image:
    :param filter: 频域变换函数
    :return:
    """
    fftImg = np.fft.fft2(image) #对图像进行傅里叶变换
    fftImgShift = np.fft.fftshift(fftImg)#傅里叶变换后坐标移动到图像中心
    handle_fftImgShift1 = fftImgShift*filter # 对傅里叶变换后的图像进行频域变换

    handle_fftImgShift2 = np.fft.ifftshift(handle_fftImgShift1)
    handle_fftImgShift3 = np.fft.ifft2(handle_fftImgShift2)
    handle_fftImgShift4 = np.real(handle_fftImgShift3)#傅里叶反变换后取频域
    return np.uint8(handle_fftImgShift4)