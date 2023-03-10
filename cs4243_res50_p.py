import glob
import random
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# gpu setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


def seed_torch(seed=4):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
class SaveBestModel:
    def __init__(
        self, save_path = "outputs", best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.save_path = save_path
        # 别忘了创建目录
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
            print(f"create {save_path}/ dir")
            
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{self.save_path}/best_model.pth')
            
def save_model(epochs, model, optimizer, criterion, save_path='outputs'):
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{save_path}/final_model.pth')

plt.style.use('ggplot')

def save_plots(train_acc, valid_acc, train_loss, valid_loss, model_name="", optimizer='', save_path='outputs'):
    # loss
    fig = plt.figure(figsize=(7, 5))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )

    fig.suptitle('model:'+model_name+" "+optimizer, fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss.png')
    
    # accuracy
    fig = plt.figure(figsize=(7, 5))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    
    fig.suptitle('model:'+model_name+" "+optimizer, fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/accuracy.png')
    

img_names = [] # ID
targets = [] # labels

# labels
class2label = {'normal': 0, 'carrying': 1, 'threat': 2}
label2class = {0: 'normal', 1: 'carrying', 2: 'threat'}
classes = list(class2label.keys())  # keep the orders
class_int = list(class2label.values())

root_path = './FashionDataset'  # locally


"""#### Image preporcess: transform"""
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)

resize_H, resized_W = 224,224 # 如果是toTensor前，最后一个transform操作(即输入到模型的input shape)，最好是kerel size, 3/5/7,* 2的指数次方
resize = transforms.Resize([resize_H, resized_W])

from torchvision import transforms as tfs
transformations = transforms.Compose([
                                    resize,
                                    # tfs.RandomHorizontalFlip(0.2),
                                    # tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                    transforms.ToTensor(), # Tturn gray level from 0-255 into 0-1
                                    # transforms.RandomAffine(0, None, (0.7, 1.1), (0, 30),fillcolor=(127,127,127)),
                                    normalize
                                    ])  #  change 0-1 into (-1, 1)

transformations_val = transforms.Compose([
                                    resize,
                                    transforms.ToTensor(), # Tturn gray level from 0-255 into 0-1
                                    normalize
                                    ])  #  change 0-1 into (-1, 1)

class CS4243_dataset(Dataset): 
    
    
    def __init__(self, root_path ,mode = 'train', transform=None):
        
        self.transform = transform
        self.root_path = root_path
        self.mode = mode

        self.img_path_list = sorted(glob.glob(f"{self.root_path}/img/*.jpg"))
        self.split = os.path.join(self.root_path, 'split')

        self.img_index_list = list(np.loadtxt(os.path.join(self.split, f'{self.mode}.txt'), dtype=str))
        if self.mode != 'test':
            self.labels = list(np.loadtxt(os.path.join(self.split, f'{self.mode}_attr.txt'), dtype=str))
    def __getitem__(self, index):
        
        img_name = self.img_index_list[index] # 没有class_file_path
        image = Image.open(os.path.join(self.root_path, img_name))
        target = torch.tensor(np.array(self.labels[index], dtype=np.float32))

        if self.transform != None:
            image = self.transform(image)
          
        return [image, target]
       
    def __len__(self):
        return len(self.img_index_list)

my_lr= 1e-4
batch_size = 32
epochs = 150

train_dataset = CS4243_dataset(root_path,mode = 'train', transform = transformations)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle = True, pin_memory=True)

val_dataset = CS4243_dataset(root_path,mode = 'val', transform = transformations_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,num_workers=0, shuffle = False, pin_memory=True)

def train(model, train_loader, optimizer, criterion):
    model.train()
    print('===== Training =====')
    running_loss=0
    show_bt_loss_freq = 200
    running_correct = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        X_train, y_train = data
        X_train = X_train.to(device)
        y_train = y_train.to(device).long()
        optimizer.zero_grad()

        # forward pass
        outputs = model(X_train)
        # calculate the loss
        loss0 = criterion(outputs[0], y_train[:,0])
        loss1 = criterion(outputs[1], y_train[:,1])
        loss2 = criterion(outputs[2], y_train[:,2])
        loss3 = criterion(outputs[3], y_train[:,3])
        loss4 = criterion(outputs[4], y_train[:,4])
        loss5 = criterion(outputs[5], y_train[:,5])
        cur_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
        # acc
        cur_acc = 0
        for c in range(6):
            cur_acc += (outputs[c].argmax(1) == y_train[:,c]).sum().item()
        running_correct += cur_acc
        cur_loss.backward()
        optimizer.step()
        
        if (i + 1) % show_bt_loss_freq == 0:
            print(f"{i+1} batches: {i+1}-batch-avg loss {cur_loss:.6f}")
            print(cur_acc/ (26 * batch_size))

        running_loss += cur_loss.item()
    epoch_loss = running_loss / len(train_loader) # avg loss on each batch
    epoch_acc = 100. * (running_correct / (len(train_loader.dataset) * 26))
    return epoch_loss , epoch_acc

# val/test a Epoch
def validate(model, test_loader, criterion):
    model.eval() # close BN and dropout layer
    print('===== Validation =====')
    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):

            X_train, y_train = data
            X_train = X_train.to(device)
            y_train = y_train.to(device).long()
            optimizer.zero_grad()
            cur_acc = 0

            # forward pass
            outputs = model(X_train)
            # calculate the loss
            loss0 = criterion(outputs[0], y_train[:, 0])
            loss1 = criterion(outputs[1], y_train[:, 1])
            loss2 = criterion(outputs[2], y_train[:, 2])
            loss3 = criterion(outputs[3], y_train[:, 3])
            loss4 = criterion(outputs[4], y_train[:, 4])
            loss5 = criterion(outputs[5], y_train[:, 5])
            cur_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
            for c in range(6):
                cur_acc += (outputs[c].argmax(1) == y_train[:,c]).sum().item()
            running_correct += cur_acc

            running_loss += cur_loss.item()

    epoch_loss = running_loss / len(test_loader) # 样本总数/batchsize 是走完一个epoch所需的“步数”
    epoch_acc = 100. * (running_correct / (len(test_loader.dataset) * 26))

    return epoch_loss , epoch_acc

"""### initalize Model & Hyper params"""
from torchvision import models
net = models.resnet50(pretrained=True)
# for name, parameter in net.named_parameters():
#     parameter.requires_grad = False
# 写一个全连接网络
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.dp_rate = 0.1
        self.fc1 = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.ReLU(),
            # nn.Dropout(self.dp_rate),
            nn.Linear(128, 7)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.ReLU(),
            # nn.Dropout(self.dp_rate),
            nn.Linear(128, 3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.ReLU(),
            # nn.Dropout(self.dp_rate),
            nn.Linear(128, 3)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.ReLU(),
            # nn.Dropout(self.dp_rate),
            nn.Linear(128, 4)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.ReLU(),
            # nn.Dropout(self.dp_rate),
            nn.Linear(128, 6)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.ReLU(),
            # nn.Dropout(self.dp_rate),
            nn.Linear(128, 3)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        return x1, x2, x3, x4, x5, x6

fc_net = MyNet()
# 把resnet的fc层去掉
net.fc = nn.Sequential()


final_model = nn.Sequential(
        net,
        fc_net
)

# send to gpu
final_model = final_model.to(device)
mean = torch.FloatTensor(mean)
mean = mean.to(device)
std = torch.FloatTensor(std)
std = std.to(device)



criterion = nn.CrossEntropyLoss()
bs= batch_size
optimizer = torch.optim.Adam(final_model.parameters(), lr=my_lr, weight_decay=1e-5)
# optimizer = torch.optim.SGD(final_model.parameters(), lr = my_lr, momentum=0.9)

model_name = net.__class__.__name__
opti_name = optimizer.__class__.__name__ 
if resize_H != 224: # 如果resize不是默认的224
    save_path = model_name + "_" + str(resize_H) + "," + str(resized_W) + "_" + opti_name + "_ep" + str(epochs) + "_lr" + str(my_lr)
else:
    save_path = model_name + "_" + opti_name + "_ep" + str(epochs) + "_lr" + str(my_lr)
    
save_best_model = SaveBestModel(save_path=save_path)
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    
    train_epoch_loss, t_acc = train(final_model, train_loader, optimizer, criterion)
    valid_epoch_loss, v_acc = validate(final_model, val_loader, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(t_acc)
    valid_acc.append(v_acc)
    print(f"Epoch Training loss: {train_epoch_loss:.5f}, training acc: {t_acc:.3f}%")
    print(f"Epoch Validation loss: {valid_epoch_loss:.5f}, validation acc: {v_acc:.3f}%")
    # save the best final_model:
    save_best_model(
        valid_epoch_loss, epoch, final_model, optimizer, criterion
    )
    print('-'*50)

print(f'The best minimal val loss is {save_best_model.best_valid_loss}')
    
save_model(epochs, final_model, optimizer, criterion, save_path=save_path)
save_plots(train_acc, valid_acc, train_loss, valid_loss, model_name=model_name, optimizer=opti_name, save_path=save_path)
print('TRAINING COMPLETE')

