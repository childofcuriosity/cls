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
parser.add_argument('--batch_size', type=int, default=32, help='The batch size for both training and validation data loaders')
parser.add_argument('--dropout',type=float, default=0.5, help='dropout before fc')
parser.add_argument('--attention_dropout',type=float, default=0.3, help='dropout')
parser.add_argument('--classify_dropout',type=float, default=0.2, help='dropout')
parser.add_argument('--lr0',type=float, default=0.0001, help='lr')
parser.add_argument('--lr1',type=float, default=0.001, help='lr')
parser.add_argument('--num_classes', type=int, default=260, help='')
parser.add_argument('--d', type=int, default=64, help='in paper is query dimension in the second stage')
parser.add_argument('--embed_dim', type=int, default=64, help='') # 是多头一起的
parser.add_argument('--num_heads', type=int, default=8, help='')
parser.add_argument('--ffn_d', type=int, default=256, help='')
parser.add_argument('--gamma_neg',type=float, default=4, help='')
parser.add_argument('--gamma_pos',type=float, default=1, help='')



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
train_DataLoader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valid_DataLoader = DataLoader(valid_set, batch_size=args.batch_size)

import torch.nn as nn

class BinaryClassifiers(nn.Module):
    def __init__(self, d_model, num_classes, dropout):
        super(BinaryClassifiers, self).__init__()
        # self.W = nn.Parameter(torch.rand(d_model*num_classes))  # [260,512]
        self.W = nn.Parameter(torch.empty(d_model * num_classes))
        torch.nn.init.xavier_uniform_(self.W.view(d_model,num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tem=x.shape
        x = self.dropout(x.view(tem[0], -1))
        x= x*self.W
        x = x.view(tem[0], tem[-2], tem[-1])
        x = torch.sum(x, dim=2)
        return x



class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class AttentionLabel(nn.Module):
    def __init__(self):
        super(AttentionLabel, self).__init__()
        self.conv=nn.Conv2d(1664, args.d, kernel_size=1)
        # self.Q0=nn.Parameter(torch.rand(args.num_classes, args.d))  # [260,512]
        self.Q0 = nn.Parameter(torch.empty(args.num_classes, args.d))
        torch.nn.init.xavier_uniform_(self.Q0)
        self.multihead_attn_1 = nn.MultiheadAttention(args.embed_dim, args.num_heads,dropout=args.attention_dropout)
        self.multihead_attn_2 = nn.MultiheadAttention(args.embed_dim, args.num_heads,dropout=args.attention_dropout,batch_first=True)
        self.ffn=FFN(args.d,args.ffn_d,dropout=args.dropout)
        self.classify=BinaryClassifiers(args.d,args.num_classes,dropout=args.classify_dropout)

    def forward(self, x):
        # 特征图[16, 1024, 4, 6]
        x=self.conv(x)
        # [16, 512, 4, 6]
        x = x.view(x.size(0), x.size(1), -1)
        # [16, 512, 24]
        x=x.permute(0,2,1)
        # [16, 24, 512]

        # 处理Q，260token的
        Q1_1,_=self.multihead_attn_1(self.Q0,self.Q0,self.Q0)
        # [16, 260, 512]
        # 加权融合x
        x,_=self.multihead_attn_2(Q1_1.unsqueeze(0).repeat(x.size(0), 1, 1),x,x)
        # [16, 260, 512]
        x=self.ffn(x)
        # FFN结构
        # [16, 260, 512]

        # 之后就对每一个输出
        x=self.classify(x)

        return x




'''
import torch
import torch.nn as nn
x = torch.rand(16, 1024, 4, 6)
x=nn.Conv2d(1024, 512, kernel_size=1)(x)
x = x.view(x.size(0), x.size(1), -1)
Q0=nn.Parameter(torch.rand(260, 512)) 
multihead_attn = nn.MultiheadAttention(512,8)
attn_output, attn_output_weights = multihead_attn(Q0, Q0, Q0)
multihead_attn2 = nn.MultiheadAttention(512,8,batch_first=True)
a=attn_output.repeat(16, 1, 1)
attn_output_2, attn_output_weights_2 = multihead_attn2(a, x, x)
x=FFN(512,2048)(attn_output_2)
y=BinaryClassifiers(512,260)(x)
'''



class MultiLabelNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelNet, self).__init__()
        net = models.densenet169(pretrained=True,drop_rate=args.dropout)
        self.features = net.features  # 输出[16, 1664, 4, 6]
        '''丢掉了：
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        '''
        self.attention_label=AttentionLabel()

    def forward(self, x):

        x = self.features(x)
        x = F.relu(x, inplace=True)  # 这个也算是预训练模型取特征图的一部分。
        # 特征图[16, 1664, 4, 6]
        # 如果要att，在这里
        x = self.attention_label(x)
        # x = self.avgpool(x)
        # x = self.flatten(x)
        # x = self.dropout(x)
        # x = self.classifier(x)
        '''
         attention:
         先给
        '''
        return torch.sigmoid(x)



# 训练和测试代码
eps=0.0000000001
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
    train_precision = train_tp / (train_tp + train_fp + eps)
    train_recall = train_tp / (train_tp + train_fn + eps)
    train_micro_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + eps)
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
    test_precision = test_tp / (test_tp + test_fp + eps)
    test_recall = test_tp / (test_tp + test_fn + eps)
    test_micro_f1 = 2 * test_precision * test_recall / (test_precision + test_recall + eps)
    test_loss /= len(data_loader.dataset)
    return test_loss,test_micro_f1



model=MultiLabelNet(num_classes=args.num_classes)

# 定义损失函数和优化器
# criterion = nn.BCELoss()


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

        # Asymmetric Focusing
        self.xs_pos = self.xs_pos * self.targets       # mask不相干的
        self.xs_neg = self.xs_neg * self.anti_targets
        self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
        self.loss *= self.asymmetric_w

        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1)

        return _loss


criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg,gamma_pos=args.gamma_pos)

'''
分组做
'''
pretrained_params = []
other_params = []
for name, param in model.named_parameters():
    if 'features' in name:
        pretrained_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.AdamW([{'params': pretrained_params,'lr':args.lr0},
                               {'params': other_params,'lr':args.lr1}])
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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


