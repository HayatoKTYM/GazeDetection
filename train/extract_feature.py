import pandas as pd
import numpy as np

def Extract_feature(train):
    """
    人ごとにデータを読み込んでいる
    return X 入力画像
        　 y 正解ラベル
    """
    df = pd.read_csv(train+"/yes_eye_data.csv",header=None).sample(frac=1,random_state=100)#読み込んで順番shuffle
    X = df.iloc[:, 2:].values.astype(np.float32).reshape(len(df),32,96,1)
    y = df.iloc[:, 1].astype(np.int32) .values

    df   = pd.read_csv(train+"/no_eye_data.csv",header=None).sample(frac=1,random_state=100)#読み込んで順番shuffle
    X_ = df.iloc[:, 2:].values.astype(np.float32).reshape(len(df),32,96,1)
    y_ = df.iloc[:, 1].astype(np.int32) .values
    X = np.vstack((X,X_))
    y = np.append(y,y_)
    return X,y
