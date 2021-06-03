import cv2 as cv
import numpy as np

def main():
    #Load the image
    img = cv.imread("a1.png")

    #Define the images for all function each
    img_area, img_box, img_major, img_minor, img_ecc = getImages(img)

    #Convert the image in grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Apply the thresholding operation
    ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY)

    #Draw contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Area Calculation.
    getAreas(img_area, contours)
    # Box Drawing.
    drawBoxes(img_box, contours)
    # Show Major Axis.
    getMajorAxis(img_major, contours)
    # Show Minor Axis.
    getMinorAxis(img_minor, contours)
    #Calculate eccentricity
    getEccentricity(img_ecc, contours)

    cv.waitKey(0)
    cv.destroyAllWindows()

def getImages(img):
    return img.copy(), img.copy(), img.copy(), img.copy(), img.copy()

def getAreas(ref_img, contours):
    for i in range(len(contours)):
        img_w, img_h = ref_img.shape[:2]
        cv.putText(ref_img, "Image Area (Px): " + str(img_w * img_h), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        x, y, w, h = cv.boundingRect(contours[i])
        area = cv.contourArea(contours[i])
        cv.putText(ref_img, str(area), (x + 3, y + 16), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("Area Func.", ref_img)

def drawBoxes(ref_img, contours):
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(ref_img, [box], 0, (200, 0, 0), 2)
        cv.imshow("Drawing Boxes Func.", ref_img)

def getMajorAxis(ref_img, contours):
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        ellipse = cv.fitEllipse(contours[i])
        cv.ellipse(ref_img, ellipse, (0, 255, 0), 2)
        cv.putText(ref_img, str(round(ellipse[1][1])), (int(rect[0][0]), int(rect[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("Major Axis Func.", ref_img)

def getMinorAxis(ref_img, contours):
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        ellipse = cv.fitEllipse(contours[i])
        cv.ellipse(ref_img, ellipse, (0, 255, 0), 2)
        cv.putText(ref_img, str(round(ellipse[1][0])), (int(rect[0][0]), int(rect[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("Minor Axis Func.", ref_img)

def getEccentricity(ref_img, contours):
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        ellipse = cv.fitEllipse(contours[i])
        cv.ellipse(ref_img, ellipse, (0, 255, 0), 2)
        major = round(ellipse[1][1])
        minor = round(ellipse[1][0])
        a = np.square((minor/2))
        b = np.square((major/2))
        ecc = np.sqrt(1-(a/b))
        cv.putText(ref_img, str(ecc), (int(rect[0][0]), int(rect[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,cv.LINE_AA)
        cv.imshow("Eccentricity Func.", ref_img)

if __name__ == '__main__':
    main()
