from keras import layers,callbacks,optimizers
from keras.models import Model,Sequential
from Cell_img import Cell_img
import numpy as np

def buildCNN(img_w, img_h):
    model = Sequential()
    #convs
    model.add(layers.Conv2D(filters=8, kernel_size=2,padding='same',activation='relu',input_shape=(img_h,img_w,3)))
    model.add(layers.MaxPooling2D(pool_size=2,padding='same'))
    model.add(layers.Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, padding='same'))
    model.add(layers.Conv2D(filters=4, kernel_size=2, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, padding='same'))
    model.add(layers.Conv2D(filters=1, kernel_size=2, padding='same', activation='relu'))
    #output
    model.add(layers.core.Flatten())
    model.add(layers.core.Dense(units=4, activation='relu'))
    model.add(layers.core.Dense(units=1, activation='sigmoid'))
    model.summary()
    return model

def buildNN(feature_len):
    model = Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(64,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

if __name__=='__main__':
    img_w = 100
    img_h = 100
    Ci = Cell_img(img_w, img_h)
    Ci.load(filepath="cell_images/")
    Ci.divide_train_val(train_ratio=0.7)

    # training params
    train_sample_num = 15000
    val_sample_num = 1000
    n_epochs = 150
    batch_size = 5000
    lr = 0.001
    use_feature = 1


    # training data
    x_train = np.zeros(shape=(train_sample_num, img_h , img_w , 3))
    x_feature_train = np.zeros(shape=(train_sample_num, 64))
    y_train = np.zeros(shape=(train_sample_num, 1), dtype=np.uint8)
    for i in range(train_sample_num):
        x_train[i] = Ci.get_train_img_Xi(i)
        x_feature_train[i] = (Ci.get_train_feature_Xi(i))
        y_train[i] = (Ci.get_train_Yi(i))

    # validation data
    x_val = np.zeros(shape=(val_sample_num, img_h , img_w , 3))
    x_feature_val = np.zeros(shape=(val_sample_num, 64))
    y_val = np.zeros(shape=(val_sample_num, 1), dtype=np.uint8)
    for i in range(val_sample_num):
        x_val[i] = Ci.get_val_img_Xi(i)
        x_feature_val[i] = (Ci.get_val_feature_Xi(i))
        y_val[i] = (Ci.get_val_Yi(i))



    if use_feature==0:
        # build model
        model  = buildCNN(img_w=img_w,img_h=img_h)
        opt = optimizers.Adam(lr=0.1)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        mc = callbacks.ModelCheckpoint(mode='auto', filepath='Cell_img_nonefeature.h5', monitor='val_acc',
                             save_best_only='True',
                             save_weights_only='True', verbose=1)
        history = model.fit(x=x_train, y=y_train, batch_size=1000,epochs=50,
                  verbose=1,callbacks=[mc],
                  validation_data=(x_val,y_val))
    else:
        # build model
        model = buildNN(feature_len=64)
        opt = optimizers.Adam(lr=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        mc = callbacks.ModelCheckpoint(mode='auto', filepath='Cell_img_feature.h5', monitor='val_acc',
                                       save_best_only='True',
                                       save_weights_only='True', verbose=1)
        history = model.fit(x=x_feature_train, y=y_train, batch_size=batch_size, epochs=n_epochs,
                            verbose=1, callbacks=[mc],
                            validation_data=(x_feature_val, y_val))

        acc_list = np.zeros(shape=(n_epochs,5))
        acc_list[:,0] = range(1,n_epochs+1)
        acc_list[:,1] = history.history['loss']
        acc_list[:,2] = history.history['acc']
        acc_list[:,3] = history.history['val_loss']
        acc_list[:,4] = history.history['val_acc']
        np.savetxt('eval/NN_history.txt',acc_list)
        '''
        np.savetxt('val_loss.txt', history.history['val_loss'])
        np.savetxt('val_acc.txt', history.history['val_acc'])
        np.savetxt('loss.txt', history.history['loss'])
        np.savetxt('acc.txt', history.history['acc'])'''

