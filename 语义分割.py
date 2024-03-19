import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 定义U-Net的层
        # 编码器层
        self.encoder1 = self.contracting_block(1, 64)
        self.encoder2 = self.contracting_block(64, 128)
        # 此处可以添加更多的编码器层

        # 解码器层
        self.decoder1 = self.expansive_block(128, 64, 64)
        # 此处可以添加更多的解码器层

        self.final_layer = self.final_block(64, 1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channel, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        return block

    def forward(self, x):
        # 应用编码器层
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        # 此处可以添加更多的编码器层应用

        # 应用解码器层
        d1 = self.decoder1(x2)
        # 此处可以添加更多的解码器层应用

        # 最后的层
        final = self.final_layer(d1)
        return final


# 实例化模型、优化器和损失函数
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# 加载数据
train_loader = DataLoader(YourDataset(), batch_size=32, shuffle=True)

# 训练过程
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# test_loader = DataLoader(YourTestDataset(), batch_size=32, shuffle=False)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        outputs = model(inputs)
        # 这里可以添加评估代码，例如计算准确率
