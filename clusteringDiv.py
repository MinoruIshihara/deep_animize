import os
import glob

import numpy
import cv2
from PIL import Image as PILImage

from matplotlib import pyplot

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

path = os.getcwd() + '\\images'

IMAGE_W = 286
IMAGE_H = 286

def pil2CV(Img):
    imgArr = numpy.array(Img, dtype = numpy.uint8)
    return cv2.cvtColor(imgArr, cv2.COLOR_RGB2BGR)

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compensateGamma(Img):
    gamma = 1 / numpy.sqrt(Img.mean()) * 13
    g_table = numpy.array([((i / 255.0) ** (1 / gamma)) * 255 for i in numpy.arange(0, 256)]).astype("uint8")
    Img = cv2.LUT(Img, g_table)
    return Img

def compensateV(Img):
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    for i, row in enumerate(Img):
        for j, pix in enumerate(row):
            Img[i][j][1] = 255
            Img[i][j][2] = 128 + (Img[i][j][2] - 128) * 0.8
    Img = cv2.cvtColor(Img, cv2.COLOR_HSV2BGR)
    return Img

def bilateralFilterRepetition(Img, kSize, sigma1, sigma2, iteration):
    for i in range(iteration):
        Img = cv2.bilateralFilter(Img, kSize, sigma1, sigma2)
    return Img

def laplasianFilter(Img):
    edgeKernel = numpy.array([[0,-1,0], [-1,5,-1], [0,-1,0]], numpy.float32)
    Img = cv2.filter2D(Img, -1, edgeKernel)
    return Img

def k_means(Img, k):
    shape = Img.shape
    Img = Img.reshape((-1, 3))
    Img = numpy.float32(Img)
    Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    Img, Label, Center = cv2.kmeans(Img, k, None, Criteria,
                                        10, cv2.KMEANS_PP_CENTERS)
    Center = numpy.uint8(Center)
    Img = Center[Label.flatten()]
    Img = Img.reshape(shape)
    return (Img, Center)

def x_means(Img):
    Img = Img.reshape([IMAGE_W*IMAGE_H,3])
    x_meansCenter = kmeans_plusplus_initializer(Img, 4).initialize()
    imgX_means = xmeans(data = Img, initial_centers = x_meansCenter, kmax = 20, core = True)
    imgX_means.process()
    centerColors = numpy.array(imgX_means.get_centers(), dtype = 'uint8')
    for (x_means_cluster, x_means_center) in zip(imgX_means.get_clusters(), centerColors):
        for pixel in x_means_cluster:
            Img[pixel] = x_means_center
    Img = Img.reshape([IMAGE_W,IMAGE_H,3])
    return Img, centerColors

def segmentation(Img):
    Img = compensateGamma(Img)
    mImg = cv2.medianBlur(Img, 7)
    qImg = k_means(mImg, 6)[0]
    return qImg

def maskImg(Img, qImg, colors):
    maskedImgs = []
    for maskColor in colors:
        imgMask = cv2.inRange(qImg, maskColor, maskColor)
        maskedImgs.append(cv2.bitwise_and(Img, Img, mask=imgMask))
    
    return maskedImgs

def meanColor(img):
    backGroundColor = [0, 0, 0]
    clipImg = img[(img != backGroundColor).all(axis = 2)]
    return numpy.mean(clipImg, axis=0)

def replaceColorMean(img, qImg, colors):
    colorMeans = []
    maskedImgs = maskImg(img, qImg, colors)
    for maskedImg in maskedImgs:
        colorMeans.append(meanColor(maskedImg))
    for (color, colorMean) in zip(colors, colorMeans):
        qImg[numpy.where((qImg == color).all(axis = 2))] = colorMean
    return qImg

def segmentation1(Img):
    Img = compensateGamma(Img)
    Img = compensateV(Img)
    mImg = cv2.medianBlur(Img, 5)
    qImg, centers = x_means(mImg)

    return qImg, centers

def segmentation2(Img):
    PIC_NUM = 3
    Img = compensateGamma(Img)
    Img = compensateV(Img)    
    mImg = cv2.medianBlur(Img, 5)
    kmeansImg = k_means(mImg, 4)
    qImg = kmeansImg[0]
    imgCenters = kmeansImg[1]

    return qImg, imgCenters

def makeDataset():
    inputFiles = glob.glob(path + '\\content256\\*.jpg')
    for imagePath in inputFiles:
        fileName = os.path.basename(imagePath).split('.')[0]
        inputImage = cv2.imread(imagePath)
        segImg, centerColors = segmentation1(inputImage)
        segImg = replaceColorMean(inputImage, segImg, centerColors)
        outputImage = cv2.hconcat([inputImage, segImg])
        cv2.imwrite(path + '\\segmentation\\_' + fileName + ".jpg", outputImage)

def trimImg(imgWidth, imgHeight):
    inputFiles = glob.glob(path + '\\contents\\square\\*.JPG')
    print('Input files path' + path + '\\contents\\square\\*.JPG')
    for fileNum, imagePath in enumerate(inputFiles):
        fileName = os.path.basename(imagePath).split('.')[0]
        inputImage = cv2.imread(imagePath)
        rawImgH, rawImgW = (inputImage.shape[0], inputImage.shape[1])
        if rawImgW > rawImgH:
            trimImg = inputImage[ : , (rawImgW - rawImgH) // 2 : (rawImgW - rawImgH) // 2 + rawImgH ]
        else:
            trimImg = inputImage[ (rawImgH - rawImgW) // 2 : (rawImgH - rawImgW) // 2 + rawImgH , : ]
        trimImg = cv2.resize(trimImg, (imgWidth, imgHeight))
        cv2.imwrite(path + '\\content256\\' + fileName + ".jpg", trimImg)

if __name__ == '__main__':
    print('running.....')
    trimImg(IMAGE_W, IMAGE_H)
    makeDataset()