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
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='The directory to save checkpoint files')
parser.add_argument('--train_set', type=str, default='train_set.pt', help='train set path')
parser.add_argument('--valid_set', type=str, default='valid_set.pt', help='train set path')
parser.add_argument('--batch_size', type=int, default=16, help='The batch size for both training and validation data loaders')
parser.add_argument('--model',type=str, default='models.resnet18', help='model design')
parser.add_argument('--dropout',type=float, default=0, help='dropout before fc')
parser.add_argument('--lr',type=float, default=0.001, help='lr')
parser.add_argument('--test')

# models.resnet18
args = parser.parse_args()
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)



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
        if 'densenet' in args.model:
            net = eval(args.model)(pretrained=True,drop_rate=args.dropout)
        else:
            net = eval(args.model)(pretrained=True)
        modules = list(net.children())[:-1]  # 去掉最后一个全连接层
        self.features = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=args.dropout)  # 添加dropout操作
        self.classifier = nn.Linear(list(net.children())[-1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        if 'densenet' in args.model:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = self.dropout(x)  # densenet在里面对应位置有dropout
        x = x.view(x.size(0), -1)
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

eps = 0.00000001

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=eps))+self.anti_targets * torch.log(self.xs_neg.clamp(min=eps))

        scale0=self.loss.sum()

        # Asymmetric Focusing
        self.xs_pos = self.xs_pos * self.targets       # mask不相干的
        self.xs_neg = self.xs_neg * self.anti_targets
        self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
        self.loss *= self.asymmetric_w

        scale1=self.loss.sum()
        _loss = - self.loss.sum() / (x.size(0)*y.size(1)) * scale0 / (scale1+eps)

        return _loss


criterion = AsymmetricLossOptimized(gamma_neg=1,gamma_pos=1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
if os.path.isfile(os.path.join(args.checkpoint_dir, 'best_model.pt')):
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt')))
    optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'optimizer.pt')))
    scheduler.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'scheduler.pt')))
    print('Loaded state from checkpoint directory: {}'.format(args.checkpoint_dir))
else:
    print('Checkpoint files not found in {}, start training from scratch.'.format(args.checkpoint_dir))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir,'logs'))

early_stop_patience = 10  # 如果验证集的性能在连续 early_stop_patience 轮中都没有提升，则停止训练
if args.test:
    early_stop_patience = 30  #为了测试时到那里记录
stop_counter = 0
best_micro_f1=0

for epoch in range(100):
    train_loss,train_micro_f1 = train(model, train_DataLoader, criterion, optimizer, device)
    test_loss,test_micro_f1 = test(model, valid_DataLoader, criterion, device)
    scheduler.step(test_micro_f1)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Micro_F1', train_micro_f1, epoch)
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Micro_F1', test_micro_f1, epoch)
    print('epoch {}:\t train_loss:{:.4f}\t train_micro_f1:{:.4f}\t test_loss:{:.4f}\t test_micro_f1:{:.4f}'.format(epoch+1,train_loss,train_micro_f1,test_loss,test_micro_f1))
    if test_micro_f1 > best_micro_f1:
        best_micro_f1 = test_micro_f1
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(args.checkpoint_dir, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(args.checkpoint_dir, 'scheduler.pt'))
        print('Saved best model and optimizer with micro_f1 of {:.4f}'.format(best_micro_f1))
        stop_counter = 0
    else:
        stop_counter += 1
        if stop_counter == early_stop_patience:
            print('Early stopping at epoch {}: test_micro_f1 has not improved in {} epochs.'.format(epoch+1, early_stop_patience))
            break

writer.close()


