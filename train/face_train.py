__author__ = 'Hayato Katayama'
__date__ = '20190921'
"""
視線推定プログラム
label 0 look robot
        1 not look robot

入力は顔画像(96*96)
FaceTrain関数 　　　このプログラムを実行する時に使用
TimeFaceTrain関数　アクション推定など複数画像を系列として処理したい時に使用(import用)

"""
import keras
from keras.layers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))


def FaceTrain( input_img = Input(shape=(96,96,1)),freeze=False):
    x = Convolution2D(32,5,activation='relu',
                      #padding='same'
                     )(input_img)
    x = BatchNormalization()(x)
    x = Convolution2D(32,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Convolution2D(32,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs= input_img,outputs=y)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-03),
                  #optimizer = keras.optimizers.SGD(1e-02),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:-5]:
            layer.trainable = False
    return model

def TimeFaceTrain( input_img = Input(shape=(10,32,96,1)),freeze=False):
    x = TimeDistributed(Convolution2D(32,5,activation='relu'))(input_img)
    x = TimeDistributed(Convolution2D(32,5,activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Convolution2D(32,5,activation='relu'))(x)
    x = TimeDistributed(Convolution2D(32,5,activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Flatten())(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32,activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs= input_img,outputs=y)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-04),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:-5]:
            layer.trainable = False
    return model

def load_image(paths,gray_flag=0):
    #pathを受け取って画像を返す
    img_feature = []
    for path in paths:
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if x is None:
            x = np.array([0]*96*96)
        x = x.reshape(96,96,1)
        img_feature.append(x / 255.0)
    return np.array(img_feature,dtype=np.float32)

def extract_img(folder_path :str = '', label : str = 'yes'):
    """
    param: folder path
    param: label (yes[look]  or no[not look])
    return 画像配列(None,96,96,1), ラベル(None,)
    """
    img_paths = glob.glob(folder_path + '/*png')
    img_feature = load_image(img_paths)
    label = np.zeros(len(img_feature)) if label == 'yes' else np.ones(len(img_feature))
    return img_feature, label

def shuffle_samples(X, y):
    """
    X,y をindexの対応を崩さずにshuffleして返す関数
    param: X,y
    return X,y
    """
    assert len(X) == len(y), print('data length inccorrect')
    zipped = list(zip(X, y))
    np.random.shuffle(zipped)
    X_result, y_result = zip(*zipped)
    return np.array(X_result,dtype=np.float32), np.array(y_result,dtype=np.int32)    # 型をnp.arrayに変換


import glob
import argparse
import pandas as pd
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import KFold
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--out', '-o', default='./result/face',
                        help='Directory to output the result')
    parser.add_argument('--input', type=str, default='../extraction/face')

    args = parser.parse_args()

    yes_face_files = sorted(glob.glob(args.input + 'yes/*')) #人ごとの画像フォルダ
    no_face_files = sorted(glob.glob(args.input + 'no/*')) #人ごとの画像フォルダ
    #del yes_face_files[15:19]
    #del no_face_files[15:19]
    val_cnt = 0
    kf = KFold(n_splits = 10)
    test_Acc = []
    
    for train_index, test_index in kf.split(yes_face_files[:]):
        X_train,X_val,X_test=[],[],[]
        y_train,y_val,y_test=[],[],[]
        print(train_index,test_index)
        ####
        print('## making training dataset ##')
        for i in train_index[:-3]:
            
            try:
                feature, label = extract_img(yes_face_files[i],label='yes')
                X_train = np.append(X_train, feature, axis = 0) if len(X_train) != 0 else feature
                y_train = np.append(y_train , label)
                feature, label = extract_img(no_face_files[i],label='no')
                X_train = np.append(X_train, feature, axis = 0) if len(X_train) != 0 else feature
                y_train = np.append(y_train , label)
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue
        ####
        print('## making validation dataset ##')
        for i in train_index[-3:]:
            try:
                
                feature, label = extract_img(yes_face_files[i],label='yes')
                X_val = np.append(X_val, feature, axis = 0) if len(X_val) != 0 else feature
                y_val = np.append(y_val , label)
                feature, label = extract_img(no_face_files[i],label='no')
                X_val = np.append(X_val, feature, axis = 0) if len(X_val) != 0 else feature
                y_val = np.append(y_val , label)
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue
        ####
        print('## making test dataset ##')
        for i in test_index:
            try:
                
                feature, label = extract_img(yes_face_files[i],label='yes')
                X_test = np.append(X_test, feature, axis = 0) if len(X_test) != 0 else feature
                y_test = np.append(y_test , label)
                feature, label = extract_img(no_face_files[i],label='no')
                X_test = np.append(X_test, feature, axis = 0) if len(X_test) != 0 else feature
                y_test = np.append(y_test , label)
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue

        print('training size:',len(X_train),'validation size:',len(X_val),'test size:',len(X_test))
        print('label ',Counter(y_train))
        
        #Training phase ##
        model = FaceTrain()
        #model.summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        model_save = keras.callbacks.ModelCheckpoint(filepath=args.out + "weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5", monitor='val_loss',save_weights_only=True)
        hist = model.fit(X_train,y_train,
              epochs=50,
              batch_size=128,
              callbacks = [
                       early_stopping,
                       model_save
                              ],
              validation_data = (X_val,y_val),
              validation_split=0.25)
        acc = model.evaluate(X_test,y_test)[1]
        print('Accuracy:',acc)
        test_Acc.append(acc)

    print("##mean Accuracy:",np.mean(test_Acc))
