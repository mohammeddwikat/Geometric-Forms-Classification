import cv2
import numpy as np
import matplotlib.pyplot as plt

def addPath(arr, path):
    for i in range(len(arr)):
        test = path
        test += arr[i]
        arr[i] = test
    return arr

#Test Cases:
# https://drive.google.com/drive/folders/1UExR9ohAdutC0vXscmNyLvZwxiOAr1Kg?usp=sharing

str1 = "TestCases/Rectangle/"
rectangles = ["t1.png", "t7.png"]
rectangles = addPath(rectangles, str1)

str1 = "TestCases/Triangle/"
triangles = ["t2.png", "t8.png"]
triangles = addPath(triangles, str1)

str1 = "TestCases/Circle/"
circles = ["t3.png", "t9.png"]
circles = addPath(circles, str1)

str1 = "TestCases/TreeClass1/"
tree1 = ["t5.png"]
tree1 = addPath(tree1, str1)

str1 = "TestCases/TreeClass2/"
tree2 = ["t6.png", "t14.png"]
tree2 = addPath(tree2, str1)

str1 = "TestCases/All/"
all = ["t13.png", "t10.png", "t11.png", "t12.png"]
all = addPath(all, str1)

arr = tree2 + all

for i in range(len(arr)):
    originalImage = cv2.imread(arr[i])
    image = cv2.imread(arr[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    ret, thresh = cv2.threshold(img, 220, 255, 1)
    edges = cv2.Canny(thresh, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    myList = []
    counter = 0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 100:
            counter += 1
            myList.append(contours[i])

    for i in range(len(myList)):
        x, y, w, h = cv2.boundingRect(myList[i])
        croppedimg = originalImage[y:y + h, x:x + w]
        epsilon = 0.1 * cv2.arcLength(myList[i], True)
        approx = cv2.approxPolyDP(myList[i], 0.03 * cv2.arcLength(myList[i], True), True)
        n = (len(approx))

        if n == 7 or n == 5:
            approx = cv2.approxPolyDP(myList[i], 0.01 * cv2.arcLength(myList[i], True), True)

        n = (len(approx))

        if n == 3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(originalImage, 'Triangle', (x+(w//3), y+(h//5)), font, 0.6, (0, 230, 0), 1, cv2.LINE_AA)
            #print("Triangle")

        elif n == 4:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(originalImage, 'Rectangle', (x+(w//3), y+(h//6)), font, 0.6, (0, 230, 0), 1, cv2.LINE_AA)
            #print("Rectangle")

        elif n == 7:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(originalImage, 'TreeClass2', (x+(w//3), y+(h//6)), font, 0.6, (0, 230, 0), 1, cv2.LINE_AA)
            #print("Tree Class 2")

        elif n == 8:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(originalImage, 'Circle', (x+(w//3), y+(h//6)), font, 0.6, (0, 230, 0), 1, cv2.LINE_AA)
            #print("Circle")

        elif n > 8:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(originalImage, 'TreeClass1', (x+(w//3), y+(h//6)), font, 0.6, (0, 230, 0), 1, cv2.LINE_AA)
            #print("Tree Class 1")

        #cv2.imshow("cropped", croppedimg)
        #cv2.waitKey(0)

    #matr = np.concatenate((img, thresh), axis=1)
    #plt.imshow(matr, cmap="gray")
    #plt.show()

    plt.imshow(originalImage)
    plt.show()

