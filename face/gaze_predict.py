__author__ = 'Hayato Katayama'
__date__ = '20190927'

"""
視線推定を行うプログラム
"""
import sys
sys.path.append('..')
from img_process import *
from cnn_model import GazeTrain

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import glob
from collections import Counter
from sklearn.model_selection import KFold


def train(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # 学習ループ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            if (epoch == 0) and (phase == 'train'):  # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
                continue
            cnt = 0
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                out  = net(inputs)  # 順伝播

                loss = criterion(out, labels)  # ロスの計算
                _, preds = torch.max(out, 1)  # ラベルを予測

                if phase == 'train':  # 訓練時はバックプロパゲーション
                    optimizer.zero_grad()  # 勾配の初期化
                    loss.backward()  # retain_graph=True) # 勾配の計算
                    optimizer.step()  # パラメータの更新

                epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #print('')

    y_true, y_pred = np.array([]), np.array([])
    for inputs, labels in dataloaders_dict['test']:
        inputs = inputs.to(device)

        out = net(inputs)  # 順伝播
        # loss = criterion(out, labels) # ロスの計算
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())

    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    torch.save(net.state_dict(), './gaze_model_64dim_face.pth')

def main():
    # 画像pathの読み込み
    folder = '/mnt/aoni02/katayama/short_project/proken2018_A/data/face/'
    yes_face_files = sorted(glob.glob(folder + 'yes/*'))[:]  # 人ごとの画像フォルダ
    no_face_files = sorted(glob.glob(folder + 'no/*'))[:]  # 人ごとの画像フォルダ
    x = y = Acc = []
    val_cnt = 0
    kf = KFold(n_splits=15)
    test_Acc = []

    for train_index, test_index in kf.split(yes_face_files[:]):
        img_path = []
        img_val_path = []
        img_test_path = []
        print(train_index, test_index)
        for i in train_index:
            try:
                img_path += glob.glob(yes_face_files[i] + '/*png')
                img_path += glob.glob(no_face_files[i] + '/*png')
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue
        for i in test_index:
            try:
                img_val_path += glob.glob(yes_face_files[i] + '/*png')
                img_val_path += glob.glob(no_face_files[i] + '/*png')               
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue

        print('training size:', len(img_path), 
        'validation size:', len(img_val_path), 'test size:', len(img_test_path))

        train_data = EyeDataset(img_path, ImageTransform(), phase='train')
        val_data = EyeDataset(img_val_path, ImageTransform(), phase='val')

        batch_size = 64
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)

        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False)

        # 辞書オブジェクトにまとめる
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": val_dataloader}
        net = GazeTrain()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion, optimizer=optimizer, num_epochs=20)
        #break


if __name__ == '__main__':
    main()
