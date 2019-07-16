import random
import cv2
from Feature_Detect import select_feature,detect_feature
from glob import glob
import numpy as np

class Cell_img:
    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h

    def load(self, filepath):
        """
        img1: parasitized cell imgs
        img0: uninfected cell imgs
        :param filepath: ends with '/', like "f://dataset/cell_images/"
        :return:
        """
        self.img1_list = glob(filepath + 'Parasitized/*')
        self.img0_list = glob(filepath + 'Uninfected/*')
        self.img1_num = len(self.img1_list)
        self.img0_num = len(self.img0_list)

    def divide_train_val(self, train_ratio):
        """
        divide the imgs into training and validation
        :param train_ratio: value from 0-1, like 0.7
        :return:
        """
        #divide
        self.train_list = self.img1_list[0:int(train_ratio * self.img1_num)]
        self.train_list[len(self.train_list):len(self.train_list)] = self.img0_list[0:int(train_ratio * self.img0_num)]
        self.train_num = len(self.train_list)

        self.val_list = self.img1_list[int(train_ratio * self.img1_num):]
        self.val_list[len(self.val_list):len(self.val_list)] = self.img0_list[int(train_ratio * self.img0_num):]
        self.val_num = len(self.val_list)

        #shuffle
        random.seed(1)
        random.shuffle(self.train_list)
        random.shuffle(self.val_list)

        #label
        self.train_label = []
        self.val_label = []
        for img in self.train_list:
            if "Parasitized" in img:
                self.train_label.append(1)
            else:
                self.train_label.append(0)

        for img in self.val_list:
            if "Parasitized" in img:
                self.val_label.append(1)
            else:
                self.val_label.append(0)

    def get_train_Xi(self,i):
        img = cv2.imread(self.train_list[i])
        img_resized = cv2.resize(src=img, dsize=(self.img_w,self.img_h))
        img_reshaped = np.reshape(img_resized,(self.img_w*self.img_h*3))
        #normalize
        img_rescaled = (img_reshaped-np.mean(img_reshaped))/np.var(img_reshaped)
        return img_rescaled
    def get_train_Yi(self,i):
        return self.train_label[i]

    def get_val_Xi(self,i):
        img = cv2.imread(self.val_list[i])
        img_resized = cv2.resize(src=img, dsize=(self.img_w, self.img_h))
        img_reshaped = np.reshape(img_resized, (self.img_w * self.img_h * 3))
        #normalize
        img_rescaled = (img_reshaped-np.mean(img_reshaped))/np.var(img_reshaped)
        return img_rescaled
    def get_val_Yi(self,i):
        return self.val_label[i]

    def get_train_feature_Xi(self,i):
        img = cv2.imread(self.train_list[i])
        des = detect_feature(img)
        des_s = select_feature(des)
        des_rescaled = (des_s-np.mean(des_s))/np.var(des_s)
        return des_rescaled
    def get_val_feature_Xi(self,i):
        img = cv2.imread(self.val_list[i])
        des = detect_feature(img)
        des_s = select_feature(des)
        des_rescaled = (des_s - np.mean(des_s)) / np.var(des_s)
        return des_rescaled

    def get_train_img_Xi(self,i):
        img = cv2.imread(self.train_list[i])
        img_resized = cv2.resize(src=img, dsize=(self.img_w, self.img_h))
        return img_resized
    def get_train_binary_Yi(self,i):
        label = self.train_label[i]
        if label==1:
            return [1,0]
        else:
            return [0,1]
    def get_val_img_Xi(self,i):
        img = cv2.imread(self.val_list[i])
        if img is None:
            img = np.ones(shape=(self.img_h,self.img_w,3))
        img_resized = cv2.resize(src=img, dsize=(self.img_w, self.img_h))
        return img_resized
    def get_val_binary_Yi(self,i):
        label = self.val_label[i]
        if label==0:
            return [1,0]
        else:
            return [0,1]
'''
Example
Ci = Cell_img(100,100)
Ci.load(filepath="E://dataset/cell_images/")
Ci.divide_train_val(train_ratio=0.7)
'''