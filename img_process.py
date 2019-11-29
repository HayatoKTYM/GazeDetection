import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image


class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。
    Attributes
    ----------
    
    """

    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ColorJitter(), #色相の変化
                transforms.RandomAffine(5), #角度変更
                #transforms.Pad(5),#周囲を0padding
                transforms.Grayscale(), #白黒
                transforms.ToTensor(),  # テンソルに変換
                
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),  # テンソルに変換
                
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)





class EyeDataset(data.Dataset):
    """
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)

        # ラベルを数値に変更する
        label = 0 if 'yes' in img_path else 1

        return img_transformed, label