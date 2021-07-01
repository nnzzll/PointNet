import os
import glob
import h5py
import time
import torch
import torch.optim as optim
from torch.backends import cudnn
from model import PointNet
from utils import PointNetDataset, PointNetLoss
from torch.utils.data import DataLoader

if __name__ == '__main__':
    DATA_DIR="ModelNet10"
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BEST_WEIGHTS = None
    MAX_VAL = 0
    cudnn.benchmark = True if DEVICE=="cuda" else False

    folders = glob.glob(os.path.join(DATA_DIR, "[!R]*"))
    class_map = {}
    for i,folder in enumerate(folders):
        class_map[i] = folder.split("\\")[-1]
    
    # Read Data
    with h5py.File("ModelNet10.h5", "r") as f:
        print(f.keys())
        train_points = f["train_x"][:]
        train_labels = f["train_y"][:]
        test_points = f["test_x"][:]
        test_labels = f["test_y"][:]

    train_set = PointNetDataset(train_points,train_labels)
    test_set = PointNetDataset(test_points,test_labels,False)
    train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    model = PointNet()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    
    for epoch in range(EPOCHS):
        ep_loss = 0.0
        TP = 0
        begin = time.time()
        model.train()
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            input,label = data[0].to(DEVICE),data[1].to(DEVICE)
            output,matrix_3x3,matrix_64x64 = model(input)
            loss = PointNetLoss(output,label,matrix_3x3,matrix_64x64)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            _,pred = torch.max(output,dim=1)
            TP += torch.sum(pred==label).detach().cpu().numpy()


        ep_loss /= len(train_loader)
        acc = TP/len(train_set)

        model.eval()
        TP = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                input,label = data[0].to(DEVICE),data[1].to(DEVICE)
                output,_,_ = model(input)
                _,pred = torch.max(output,dim=1)
                TP += torch.sum(pred==label).detach().cpu().numpy()
        val = TP/len(test_set)
        print("Epoch:{}/{}\tLoss:{:.6f}\tTrain:{:.2f}%\tVal:{:.2f}%\tTime:{:.3f}s".format(
            epoch+1,EPOCHS,ep_loss,acc*100,val*100,time.time()-begin))
        if val>MAX_VAL:
            MAX_VAL = val
            BEST_WEIGHTS = model.state_dict()

    state = {"net":BEST_WEIGHTS}
    torch.save(state,"model/PointNetClf-ep20.pth")