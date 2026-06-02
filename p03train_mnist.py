import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        return x @ self.weight.T + self.bias

model = nn.Sequential(
    SimpleLinear(784, 128),
    nn.ReLU(),
    SimpleLinear(128, 10)
)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model.train()
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx:4d}/938 | Loss: {loss.item():.4f}')

print('\nEpoch 1 done. Model trained on 60,000 images.')

# TEST BLOCK STARTS HERE
model.eval()
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 784)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'\nTest Accuracy: {100 * correct / total:.2f}%')
print(f'Correct: {correct}/10000')
