import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from Cell_img import Cell_img

def rdf_train_val(n_trees=10, train_sample_num = 2000, val_sample_num = 100, verbose=0):
    # training data
    x_train = np.zeros(shape=(train_sample_num, img_h * img_w * 3))
    x_feature_train = np.zeros(shape=(train_sample_num, 64))
    y_train = np.zeros(shape=(train_sample_num), dtype=np.uint8)
    for i in range(train_sample_num):
        x_feature_train[i] = (Ci.get_train_feature_Xi(i))
        y_train[i] = (Ci.get_train_Yi(i))

    # classification
    rf = RandomForestClassifier(n_estimators=n_trees, bootstrap=True)
    rf.fit(x_feature_train, y_train)

    # training metrics
    y_pred_train = rf.predict(x_feature_train)
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
    y_pred_val = rf.predict(x_feature_val)
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

    # ntrees ~ acc
    iter = 11
    acc_list = np.zeros(shape=(iter, 3))
    for i in range(iter):
        n_trees = np.power(2, i)
        acc_list[i][0] = n_trees
        acc_list[i][1:] = rdf_train_val(n_trees=n_trees,train_sample_num=10000,val_sample_num=1000)
    np.savetxt('eval/rdf_ntrees.txt', acc_list)







