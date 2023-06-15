from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import random
import numpy as np
from _1_pre_data import CustomDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='The directory to save checkpoint files')
parser.add_argument('--train_set', type=str, default='train_set.pt', help='train set path')
parser.add_argument('--valid_set', type=str, default='valid_set.pt', help='train set path')
parser.add_argument('--batch_size', type=int, default=32, help='The batch size for both training and validation data loaders')
parser.add_argument('--dropout',type=float, default=0.5, help='dropout before fc')
parser.add_argument('--attention_dropout',type=float, default=0.5, help='dropout')
parser.add_argument('--classify_dropout',type=float, default=0.2, help='dropout')
parser.add_argument('--lr',type=float, default=0.001, help='lr')

# models.resnet18
args = parser.parse_args()
# if not os.path.exists(args.checkpoint_dir):
#     os.makedirs(args.checkpoint_dir)



seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

train_set=torch.load(args.train_set)
valid_set=torch.load(args.valid_set)
train_DataLoader = DataLoader(train_set, batch_size=16, shuffle=True)
valid_DataLoader = DataLoader(valid_set, batch_size=16)

class MultiLabelNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelNet, self).__init__()
        net = models.swin_v2_b(pretrained=True,dropout=args.dropout,attention_dropout=args.attention_dropout)
        modules = list(net.children())[:-3]  # 去掉avgpool flatten classifier
        self.features = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=args.classify_dropout)  # 添加dropout操作
        self.classifier = nn.Linear(list(net.children())[-1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        # 特征图[16, 1024, 4, 6]
        # 如果要att，在这里
        x = self.avgpool(x)
        x = self.flatten(x)
        #
        x = self.dropout(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

# 训练和测试代码
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_tp=0  # label 1 predict 1
    train_fp=0  # label 0 predict 1
    train_fn=0  # label 1 predict 1
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        predicted_labels = (outputs > 0.5).type(torch.FloatTensor).to(device)
        train_tp += ((predicted_labels == 1) & (labels == 1)).sum().item()
        train_fp += ((predicted_labels == 1) & (labels == 0)).sum().item()
        train_fn += ((predicted_labels == 0) & (labels == 1)).sum().item()
    train_loss /= len(data_loader.dataset)
    train_precision = train_tp / (train_tp + train_fp)
    train_recall = train_tp / (train_tp + train_fn)
    train_micro_f1 = 2 * train_precision * train_recall / (train_precision + train_recall)
    return train_loss,train_micro_f1

def test(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_tp=0  # label 1 predict 1
    test_fp=0  # label 0 predict 1
    test_fn=0  # label 1 predict 1

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predicted_labels = (outputs > 0.5).type(torch.FloatTensor).round().type(torch.IntTensor).to(device)
            test_tp += ((predicted_labels == 1) & (labels == 1)).sum().item()
            test_fp += ((predicted_labels == 1) & (labels == 0)).sum().item()
            test_fn += ((predicted_labels == 0) & (labels == 1)).sum().item()
    test_precision = test_tp / (test_tp + test_fp)
    test_recall = test_tp / (test_tp + test_fn)
    test_micro_f1 = 2 * test_precision * test_recall / (test_precision + test_recall)
    test_loss /= len(data_loader.dataset)
    return test_loss,test_micro_f1



model=MultiLabelNet(num_classes=260)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


warmup = [0.0000001, 0.0000002, 0.0000005, 0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001]
def rule(epoch):
    return warmup[epoch]
scheduler0 = lr_scheduler.LambdaLR(optimizer, lr_lambda = rule)
scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

#
# if os.path.isfile(os.path.join(args.checkpoint_dir, 'best_model.pt')):
#     model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt')))
#     optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'optimizer.pt')))
#     scheduler.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'scheduler.pt')))
#     print('Loaded state from checkpoint directory: {}'.format(args.checkpoint_dir))
# else:
#     print('Checkpoint files not found in {}, start training from scratch.'.format(args.checkpoint_dir))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir,'logs'))

early_stop_patience = 10  # 如果验证集的性能在连续 early_stop_patience 轮中都没有提升，则停止训练
stop_counter = 0
best_micro_f1=0

for epoch in range(100):
    train_loss,train_micro_f1 = train(model, train_DataLoader, criterion, optimizer, device)
    test_loss,test_micro_f1 = test(model, valid_DataLoader, criterion, device)
    if epoch<len(warmup):
        scheduler0.step()
    else:
        scheduler1.step(test_micro_f1)
    # writer.add_scalar('Train/Loss', train_loss, epoch)
    # writer.add_scalar('Train/Micro_F1', train_micro_f1, epoch)
    # writer.add_scalar('Test/Loss', test_loss, epoch)
    # writer.add_scalar('Test/Micro_F1', test_micro_f1, epoch)
    print('epoch {}:\t train_loss:{:.4f}\t train_micro_f1:{:.4f}\t test_loss:{:.4f}\t test_micro_f1:{:.4f}'.format(epoch+1,train_loss,train_micro_f1,test_loss,test_micro_f1))
    if test_micro_f1 > best_micro_f1:
        best_micro_f1 = test_micro_f1
        # torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pt'))
        # torch.save(optimizer.state_dict(), os.path.join(args.checkpoint_dir, 'optimizer.pt'))
        # torch.save(scheduler.state_dict(), os.path.join(args.checkpoint_dir, 'scheduler.pt'))
        # print('Saved best model and optimizer with micro_f1 of {:.4f}'.format(best_micro_f1))
        stop_counter = 0
    else:
        stop_counter += 1
        if stop_counter == early_stop_patience:
            print('Early stopping at epoch {}: test_micro_f1 has not improved in {} epochs.'.format(epoch+1, early_stop_patience))
            break

# writer.close()


