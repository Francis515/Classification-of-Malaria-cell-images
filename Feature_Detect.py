import cv2
import numpy as np

def detect_feature(img, threshold=500):
    """
    use SURF to detect image features
    :param img:
    :param threshold:
    :return: feature descriptor
    """
    surf = cv2.xfeatures2d.SURF_create(threshold)
    kp, des = surf.detectAndCompute(img, None)
    if des is None:
        des = np.random.rand(1,64)
    return des

def select_feature(des):
    """
    Select the descriptor with max harr wavelet response
    :param des: descriptors
    :return: the most markable feature
    """
    s = []
    for i in range(int(des.shape[0])):
        ft = des[i]
        #s = sigma |dx| + sigma |dy| in 4x4 areas
        s.append(sum(ft[2::4])+sum(ft[3::4]))
    idx = s.index(np.max(s))
    return des[idx]

'''
img = cv2.imread('cell_images/u.png')
surf = cv2.xfeatures2d.SURF_create(500)
kp, des = surf.detectAndCompute(img, None)
img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
cv2.imwrite('allfeature_u.png',img2)
des_s = select_feature(des)
img3 = cv2.drawKeypoints(img, [kp[0]], None, (255, 0, 0), 4)
cv2.imwrite('onefeature_u.png',img3)'''