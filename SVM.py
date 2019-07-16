from Cell_img import Cell_img
import cv2
import numpy as np
import sklearn
from sklearn import linear_model,metrics,svm

def svm_train_val(train_sample_num = 2000, val_sample_num = 100, verbose=0):
    # training data
    x_train = np.zeros(shape=(train_sample_num, img_h * img_w * 3))
    x_feature_train = np.zeros(shape=(train_sample_num, 64))
    y_train = np.zeros(shape=(train_sample_num), dtype=np.uint8)
    for i in range(train_sample_num):
        x_feature_train[i] = (Ci.get_train_feature_Xi(i))
        y_train[i] = (Ci.get_train_Yi(i))

    # classifier
    svc = linear_model.SGDClassifier(loss='squared_hinge',penalty='l2',max_iter=5000,shuffle=False)
    svc.fit(x_feature_train, y_train)

    # training metrics
    y_pred_train = svc.predict(x_feature_train)
    if verbose == 1:
        print("Training metrics:")
        print(sklearn.metrics.classification_report(y_true=y_train, y_pred=y_pred_train))
    else:
        train_acc = sklearn.metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)

    # validation data
    x_val = np.zeros(shape=(val_sample_num, img_h * img_w * 3))
    x_feature_val = np.zeros(shape=(val_sample_num, 64))
    y_val = np.zeros(shape=(val_sample_num), dtype=np.uint8)
    for i in range(val_sample_num):
        x_feature_val[i] = (Ci.get_val_feature_Xi(i))
        y_val[i] = (Ci.get_val_Yi(i))

    # validation metrics
    y_pred_val = svc.predict(x_feature_val)
    if verbose == 1:
        print("Validation metrics:")
        print(sklearn.metrics.classification_report(y_true=y_val, y_pred=y_pred_val))
    else:
        val_acc = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred_val)

    return [train_acc, val_acc]

if __name__=='__main__':

    img_w = 100
    img_h = 100
    Ci = Cell_img(img_w, img_h)
    Ci.load(filepath="cell_images/")
    Ci.divide_train_val(train_ratio=0.7)

    # train_sample_num ~ acc
    iter = 15
    acc_list = np.zeros(shape=(iter, 3))
    for i in range(iter):
        n_train = 1000*(i+1)
        acc_list[i][0] = n_train
        acc_list[i][1:] = svm_train_val(train_sample_num=n_train, val_sample_num=1000)
    np.savetxt('eval/svm_ntrain.txt', acc_list)




