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

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bilateralFilterRepetition(Img, kSize, sigma1, sigma2, iteration):
    for i in range(iteration):
        Img = cv2.bilateralFilter(Img, kSize, sigma1, sigma2)
    return Img

def laplasianFilter(Img):
    edgeKernel = numpy.array([[0,-1,0], [-1,5,-1], [0,-1,0]], numpy.float32)
    Img = cv2.filter2D(Img, -1, edgeKernel)
    return Img

def compensationImg(Img):
    gamma = 1 / numpy.sqrt(Img.mean()) * 13
    g_table = numpy.array([((i / 255.0) ** (1 / gamma)) * 255 for i in numpy.arange(0, 256)]).astype("uint8")
    Img = cv2.LUT(Img, g_table)
    return Img

def compensationV(Img):
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    for i, row in enumerate(Img):
        for j, pix in enumerate(row):
            Img[i][j][1] = 255
            Img[i][j][2] = 128 + (Img[i][j][2] - 128) * 0.8
    Img = cv2.cvtColor(Img, cv2.COLOR_HSV2BGR)
    return Img

def pil2CV(Img):
    imgArr = numpy.array(Img, dtype = numpy.uint8)
    return cv2.cvtColor(imgArr, cv2.COLOR_RGB2BGR)

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
    Img = compensationImg(Img)
    mImg = cv2.medianBlur(Img, 7)
    qImg = k_means(mImg, 6)[0]
    return qImg

def maskImg(Img, qImg, colors):
    maskedImgs = []
    for maskColor in colors:
        imgMask = cv2.inRange(qImg, maskColor, maskColor)
        maskedImgs.append(cv2.bitwise_and(Img, Img, mask=imgMask))
    
    PIC_NUM = len(colors)
    #resultImgFigure = pyplot.figure(figsize = (12.0,40.0))    
    
    #for i, maskedImg in enumerate(maskedImgs):
    #    resultImgFigure.add_subplot(PIC_NUM, 1, i+1)
    #    pyplot.imshow(cv2.cvtColor(maskedImg, cv2.COLOR_BGR2RGB))
        
    return maskedImgs

def meanColor(img):
    backGroundColor = [0, 0, 0]
    clipImg = img[(img != backGroundColor).all(axis = 2)]
    #print(clipImg)
    return numpy.mean(clipImg, axis=0)

def replaceColorMean(img, qImg, colors):
    colorMeans = []
    maskedImgs = maskImg(img, qImg, colors)
    for maskedImg in maskedImgs:
        colorMeans.append(meanColor(maskedImg))
        #print(meanColor(maskedImg))
    for (color, colorMean) in zip(colors, colorMeans):
        qImg[numpy.where((qImg == color).all(axis = 2))] = colorMean
    return qImg

def segmentation720(rawImg):
    PIC_NUM = 6

    resultImgFigure = pyplot.figure(figsize = (12.0,40.0))

    resultImgFigure.add_subplot(PIC_NUM, 1, 1)
    pyplot.imshow(cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB))

    filteringImg = compensationImg(rawImg)
    resultImgFigure.add_subplot(PIC_NUM, 1, 2)
    pyplot.imshow(cv2.cvtColor(filteringImg, cv2.COLOR_BGR2RGB))

    filteringImg = bilateralFilterRepetition(filteringImg, 35, 35, 25, 10)
    filteringImg = bilateralFilterRepetition(filteringImg, 91, 35, 25, 10)
    resultImgFigure.add_subplot(PIC_NUM, 1, 3)
    pyplot.imshow(cv2.cvtColor(filteringImg, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('/content/drive/My Drive/experiment/contents/' + imageTitle + '_bilateral.jpg', filteringImg)

    filteringImg = cv2.medianBlur(filteringImg, 9)
    resultImgFigure.add_subplot(PIC_NUM, 1, 4)
    pyplot.imshow(cv2.cvtColor(filteringImg, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('/content/drive/My Drive/experiment/contents/' + imageTitle + '_median.jpg', filteringImg)

    quantizedImg = k_means(filteringImg)[0]
    resultImgFigure.add_subplot(PIC_NUM, 1, 5)
    pyplot.imshow(cv2.cvtColor(quantizedIg, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('/content/drive/My Drive/experiment/contents/' + imageTitle + '_kmeans.jpg', quantizedImg)

    quantizedImg = cv2.medianBlur(quantizedImg, 91)
    resultImgFigure.add_subplot(PIC_NUM, 1, 6)
    pyplot.imshow(cv2.cvtColor(quantizedImg, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('/content/drive/My Drive/experiment/contents/' + imageTitle + '_kmeans_median.jpg', quantizedImg)

def segmentation0(Img):
    PIC_NUM = 3
    #resultImgFigure = pyplot.figure(figsize = (8.0,16.0))

    Img = compensationImg(Img)
    #Img = compensationV(Img)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 1)
    #pyplot.imshow(BGR2RGB(Img))

    mImg = cv2.medianBlur(Img, 5)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 2)
    #pyplot.imshow(BGR2RGB(mImg))

    qImg, centers= x_means(mImg)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 3)
    #pyplot.imshow(BGR2RGB(qImg))

    return qImg, centers

def segmentation1(Img):
    #PIC_NUM = 3
    #resultImgFigure = pyplot.figure(figsize = (8.0,16.0))

    Img = compensationImg(Img)
    Img = compensationV(Img)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 1)
    #pyplot.imshow(BGR2RGB(Img))

    mImg = cv2.medianBlur(Img, 5)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 2)
    #pyplot.imshow(BGR2RGB(mImg))

    qImg, centers = x_means(mImg)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 3)
    #pyplot.imshow(BGR2RGB(qImg))

    return qImg, centers

def segmentation2(Img):
    PIC_NUM = 3
    #resultImgFigure = pyplot.figure(figsize = (8.0,16.0))

    Img = compensationImg(Img)
    Img = compensationV(Img)    

    #resultImgFigure.add_subplot(PIC_NUM, 1, 1)
    #pyplot.imshow(BGR2RGB(Img))

    mImg = cv2.medianBlur(Img, 5)

    #resultImgFigure.add_subplot(PIC_NUM, 1, 2)
    #pyplot.imshow(BGR2RGB(mImg))

    kmeansImg = k_means(mImg, 4)
    qImg = kmeansImg[0]
    imgCenters = kmeansImg[1]

    #resultImgFigure.add_subplot(PIC_NUM, 1, 3)
    #pyplot.imshow(BGR2RGB(qImg))

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
        #fileName = 'ArseniXC'
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

#imageTitle = 'content25'
#fileType = 'JPG' 

#inputFiles = glob.glob(path + 'contents/*.jpg')
#for imagePath in inputFiles:
#    fileName = os.path.basename(imagePath).split('.')[0]
#    inputImg = cv2.imread(imagePath)
#    inputImg = cv2.resize(inputImg, (256, 256))
#    cv2.imwrite(path + '/cycleGAN/dataset/train/input/' + fileName + ".jpg", inputImg)

#segImg, centers = segmentation2(inputImg)

#maskImg(inputImg, segImg, centers)
#segImg = replaceColorMean(inputImg, segImg, centers)
#pyplot.imshow(BGR2RGB(segImg))

#inputImg = cv2.imread(path + '/example/content/content41.JPG')
#qImg, imgCenters = segmentation2(inputImg)
#masks = maskImg(inputImg, qImg, imgCenters)

#resultImgFigure = pyplot.figure(figsize = (8.0,16.0))

#for i, mask in enumerate(masks):
#    cv2.imwrite(path + '/example/content/content41('+str(i)+').jpg', mask)
    #resultImgFigure.add_subplot(4, 1, i)
#    pyplot.imshow(BGR2RGB(Img))

#inputImg = cv2.imread(path + '/example/content/content41.JPG')
#qImg, imgCenters = segmentation2(inputImg)
#segImg = replaceColorMean(inputImg, qImg, imgCenters)
#cv2.imwrite(path + '/example/content41(seg2).jpg', segImg)