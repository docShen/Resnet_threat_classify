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

root_path = './data'  # locally

for class_path in classes:
    path = os.path.join(root_path, class_path)
    print('here is class:', class_path, len(os.listdir(path)))
    img_names += os.listdir(path)
    targets += [class2label[class_path]] * len(os.listdir(path))  # all labels = 0 or 1 or 2

assert len(img_names) == len(targets), f'wrong no of all imgs, {len(img_names)}, {len(targets)}'
df = pd.DataFrame({'ID': img_names, 'Label': targets})
df.head()

y = df['Label']
X = df

train_df, test_df, train_label, test_label = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.2) 

# 一定要resetindex, 给loader喂进去之前
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

"""#### Image preporcess: transform"""

mean = [0.4098, 0.4207, 0.4142]
std = [0.2732, 0.2746, 0.2714]
normalize = transforms.Normalize(mean, std)

resize_H, resized_W = 224,224 # 如果是toTensor前，最后一个transform操作(即输入到模型的input shape)，最好是kerel size, 3/5/7,* 2的指数次方
resize = transforms.Resize([resize_H, resized_W])

from torchvision import transforms as tfs
transformations = transforms.Compose([resize,
                                    tfs.RandomHorizontalFlip(0.2),
                                    tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                    transforms.ToTensor(), # Tturn gray level from 0-255 into 0-1
                                    transforms.RandomAffine(0, None, (0.7, 1.1), (0, 30),fillcolor=(127,127,127)),
                                    normalize])

transformations_test = transforms.Compose([resize,
                                    transforms.ToTensor(), # Tturn gray level from 0-255 into 0-1
                                    normalize])  #  change 0-1 into (-1, 1)

class CS4243_dataset(Dataset): 
    
    
    def __init__(self, root_path , dataframe, transform=None):
        
        self.label2class = {0:'normal', 1:'carrying', 2:'threat'}
        self.df = dataframe    
        self.transform = transform
        self.root_path = root_path
        
        self.image_paths = self.df['ID'] #image names
        self.labels = self.df['Label']


    def __getitem__(self, index):
        
        img_path = self.image_paths[index] # 没有class_file_path
        # print(img_path)
        class_path = self.label2class[self.labels[index]]
        image = Image.open(os.path.join(self.root_path, class_path, img_path))
        
        target = torch.tensor(self.labels[index])
      
        if self.transform != None:
            image = self.transform(image)
          
        return [image, target]
       
    def __len__(self):
        return len(self.df)

batch_size = 64

train_dataset = CS4243_dataset(root_path, train_df, transform = transformations)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle = True, pin_memory=True)

test_dataset = CS4243_dataset(root_path, test_df, transform = transformations_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0, shuffle = True, pin_memory=True)

def train(model, train_loader, optimizer, criterion):
    model.train()
    print('===== Training =====')
    running_loss=0
    counter=0 # batch no
    running_correct = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        X_train, y_train = data
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(X_train)
        # calculate the loss
        loss = criterion(outputs, y_train)
        running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == y_train).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
        
        # loss and accuracy for the complete epoch
        epoch_loss = running_loss / counter
        
        # acc跟当前 trained过的数量，做分母
        batch_size = X_train.shape[0]
        no_trained_examples = ((i+1)*batch_size)
        epoch_acc = 100. * (running_correct / no_trained_examples)
        
        # output batch信息
        batch_size = X_train.shape[0]
        show_bt_loss_freq = len(train_df)//batch_size//6
        if (i + 1) % show_bt_loss_freq == 0:
            print(f"{i+1} batches: {i+1}-batch-avg loss {epoch_loss:.6f}, train acc: {epoch_acc:.2f}%")
    
    assert len(train_loader) == counter, f"{len(train_loader)}, {counter}" # check no_step
    epoch_loss = running_loss / len(train_loader) # avg loss on each batch
    epoch_acc = 100. * (running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

# val/test a Epoch
def validate(model, test_loader, criterion):
    model.eval() # close BN and dropout layer
    print('===== Validation =====')
    running_loss = 0.0
    running_correct = 0
    
    counter = 0 # batch 
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            
            X_test, y_test = data
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # forward pass
            outputs = model(X_test)
            # calculate the loss
            loss = criterion(outputs, y_test)
            running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == y_test).sum().item()
        
    assert len(test_loader) == counter, f"{len(test_loader)}, {counter}" # check no_step
    epoch_loss = running_loss / len(test_loader) # 样本总数/batchsize 是走完一个epoch所需的“步数”
    epoch_acc = 100. * (running_correct / len(test_loader.dataset))
    return epoch_loss, epoch_acc

"""### initalize Model & Hyper params"""


from torchvision import models
net = models.resnet50(pretrained=True)

net.fc =  nn.Linear(net.fc.in_features,3)

# send to gpu
net = net.to(device)
mean = torch.FloatTensor(mean)
mean = mean.to(device)
std = torch.FloatTensor(std)
std = std.to(device)



criterion = nn.CrossEntropyLoss()
# my_lr=0.25 
my_lr= 1e-4
bs= batch_size
epochs = 60

optimizer = torch.optim.Adam(net.parameters(), lr=my_lr)

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
    
    train_epoch_loss, train_epoch_acc = train(net, train_loader, optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(net, test_loader, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Epoch Training loss: {train_epoch_loss:.5f}, training acc: {train_epoch_acc:.3f}%")
    print(f"Epoch Validation loss: {valid_epoch_loss:.5f}, validation acc: {valid_epoch_acc:.3f}%")
    # save the best net:
    save_best_model(
        valid_epoch_loss, epoch, net, optimizer, criterion
    )
    print('-'*50)

print(f'The best minimal val loss is {save_best_model.best_valid_loss}')
    
save_model(epochs, net, optimizer, criterion, save_path=save_path)
save_plots(train_acc, valid_acc, train_loss, valid_loss, model_name=model_name, optimizer=opti_name, save_path=save_path)
print('TRAINING COMPLETE')

