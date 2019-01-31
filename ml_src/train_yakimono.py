import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*29*29, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 29*29*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


data_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor()
])

your_datasets = datasets.ImageFolder(
        root="data",
        transform=data_transform
)
loader = torch.utils.data.DataLoader(your_datasets, batch_size=32)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    sum_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        sum_loss += loss
    print('{0}: {1}'.format(epoch, sum_loss))

torch.save(model.state_dict(), "model/yakimono_proto.pt")
