from tensorflow import keras
import tensorflow as tf

import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import KFold

from gaze_model import CNN
from extract_feature import Extract_feature

if __name__ == '__main__':
    Acc = []
    files = sorted(glob('./input/*'))
    kf = KFold(n_splits=10,shuffle=False)
    for train_idx, test_idx in kf.split(files[:]):
        print(train_idx,test_idx)
        for i in train_idx:
            try:
                if i == train_idx[0]:
                    x,y = Extract_feature(files[i])
                    X = np.array(x)
                    y_true = np.array(y)
                else:
                    x,y = Extract_feature(files[i])
                    X = np.append(X,x,axis=0)
                    y_true = np.append(y_true,y,axis=0)
            except:
                print('error file',end=":")
                print(files[i])

        for i in test_idx:
            try:
                if i == test_idx[0]:
                    x,y = Extract_feature(files[i])
                    X_val = np.array(x)
                    y_val = np.array(y)

                else:
                    x,y = Extract_feature(files[i])
                    X_val = np.append(X_val,x,axis=0)
                    y_val = np.append(y_val,y,axis=0)


            except:
                print('error file',end=":")
                print(files[i])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
        model = CNN()
        print('making model and training start ...')
        hist = model.fit(X,y_true,
                         batch_size=128,
                         epochs=50,
                         verbose = 2,
                         validation_data=(X_val, y_val),
                         callbacks=[early_stopping])



        x = model.predict(X_val).reshape(-1)
        print(model.evaluate(X_val,y_val)[1])
        y_pred = [0 if p < 0.5 else 1 for p in x]
        print(classification_report(y_true,y_pred))
