import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

model = torch.nn.Sequential(
    nn.Conv2d(1, 8, 5),  # 96 * 96 * 16-> 92 * 92 * 16
    nn.ReLU(),
    nn.MaxPool2d(2), #24 * 24 *16 -> 46 * 46 * 16    
    nn.Conv2d(8, 16,  5), # 46* 46 * 16 -> 42* 42 * 32
    nn.ReLU(),
    nn.Dropout2d(),
    Flatten(),
    nn.Linear(42 * 42 * 16, 128),
    nn.Linear(128, 2)
)

#勾配法
optimizer = optim.SGD(model.parameters(), lr=0.01)
#誤差関数
criterion = nn.CrossEntropyLoss()




from data_utils import *

def main():

    # アリとハチの画像へのファイルパスのリストを作成する
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    # Datasetを作成する
    size = 96
    #mean = (0.485, 0.456, 0.406)
    #std = (0.229, 0.224, 0.225)
    train_dataset = HymenopteraDataset(
        file_list=train_list, transform=ImageTransform(size), phase='train')

    val_dataset = HymenopteraDataset(
        file_list=val_list, transform=ImageTransform(size), phase='val')

    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    # DataLoaderを作成する
    batch_size = 32

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    num_epochs=2
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

if __name__ == '__main__':
    main()