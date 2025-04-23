import torch
from torchinfo import summary
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

from vit_pytorch import SimpleViT

from image_generator import PriceOnly, PriceAndVolume
from label_generator import LabelGenerator
from earlyStopping import EarlyStopping

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
IMG_SIZE=56
N=14
EPOCHS=120
BASE_LEARNING_RATE=3e-4
PATIENCE=4
PATCHSIZE=8
DIM=128
DEPTH=4
HEADS=4
MLPDIM=256
DECAY=1e-4

model = SimpleViT(image_size=IMG_SIZE, patch_size=PATCHSIZE,num_classes=2,dim=DIM,depth=DEPTH,heads=HEADS,mlp_dim=MLPDIM).to(device)
img_gen=PriceAndVolume(n=N)
lbl_gen=LabelGenerator(n=N)

df=pd.read_csv("finance_1d.csv")
train_df=df[:5300]
test_df=df[5300:]

train_imgs=img_gen.generate(train_df)
test_imgs=img_gen.generate(test_df)
train_labels=lbl_gen.generate(train_df)
test_labels=lbl_gen.generate(test_df)

train_size=np.array(train_imgs).shape[3]
test_size=np.array(test_imgs).shape[3]

train_inp=np.ndarray((train_size,3,IMG_SIZE,IMG_SIZE))
test_inp=np.ndarray((test_size,3,IMG_SIZE,IMG_SIZE))

for i in range(train_size):
    img=train_imgs[:,:,:,i]
    train_inp[i,0,:,:]=np.array(img)[:,:,0]
    train_inp[i,1,:,:]=np.array(img)[:,:,1]
    train_inp[i,2,:,:]=np.array(img)[:,:,2]

for i in range(test_size):
    img=test_imgs[:,:,:,i]
    test_inp[i,0,:,:]=np.array(img)[:,:,0]
    test_inp[i,1,:,:]=np.array(img)[:,:,1]
    test_inp[i,2,:,:]=np.array(img)[:,:,2]

train_tensor=torch.tensor(train_inp/255.0, dtype=torch.float32).to(device)
test_tensor=torch.tensor(test_inp/255.0, dtype=torch.float32).to(device)
train_labels_tensor=torch.tensor(train_labels, dtype=torch.int64).to(device)
test_labels_tensor=torch.tensor(test_labels, dtype=torch.int64).to(device)

dataset_train=torch.utils.data.TensorDataset(train_tensor,train_labels_tensor)
dataset_test=torch.utils.data.TensorDataset(test_tensor,test_labels_tensor)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS, pct_start=0.25, anneal_strategy='cos'
)

summary(model,input_size=(TRAIN_BATCH_SIZE,3,56,56))
earlystopping=EarlyStopping(patience=PATIENCE, verbose=True)

train_losses=[]
test_losses=[]
for i in range(EPOCHS):
    #print(torch.Tensor(np.array(img)[:,:,0]))
    train_loss=0
    test_loss=0
    model.train()
    for j, (inp, lbl) in enumerate(train_loader):
        #Train
        
        #print(f"{inp.size()} {lbl.size()}")
        optimizer.zero_grad()
        preds=model(inp)
        loss=criterion(preds, lbl)
        
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        scheduler.step()
        #print("Logits mean:", preds.mean(dim=0))
    
    model.eval()
    correct=0
    for j, (inp, lbl) in enumerate(test_loader):
        with torch.no_grad():
            #Test
            preds=model(inp)
            loss=criterion(preds, lbl)
            test_loss+=loss.item()
            probs=nn.functional.softmax(preds,dim=1)
            preds2 = torch.argmax(probs, dim=1)
            correct+=preds2.eq(lbl.view_as(preds2)).sum().item()
    train_loss/=len(dataset_train)
    test_loss/=len(dataset_test)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"EPOCH {i}, Train_Loss:{train_loss}, Test_Loss:{test_loss}, Learning Rate:{scheduler.get_lr()}, Test Accuracy:{correct / len(dataset_test)}")
    if i>12:
        #earlystopping(test_loss,model)
        earlystopping(1 - correct / len(dataset_test),model)
    if earlystopping.early_stop:
        print("Stopped Early!")
        break
    #print(train_tensor.size())
    #print(train_size//TRAIN_BATCH_SIZE+1)
    #print(preds.size())
    #print(test_tensor[0:2].size())

torch.save(model.state_dict(), f'N{N}E{i}D{datetime.today().date()}_{PATCHSIZE}_{DIM}_{DEPTH}_{HEADS}_{MLPDIM}.pth')
plt.plot(train_losses)
plt.plot(test_losses)
plt.savefig("out.png")
plt.show()
