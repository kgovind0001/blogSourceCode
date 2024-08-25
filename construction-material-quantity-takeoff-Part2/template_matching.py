import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 

threshold = 0.45

for template_path in ["symbol1.png", "symbol2.png"]: 
    img_rgb = cv.imread('cropped_image.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)

    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    cv.imwrite(f'matched_image_{template_path}',img_rgb)
