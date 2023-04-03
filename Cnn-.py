import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


train_dataset = datasets.MNIST(root='dataset/mnist',
                                        train=True,
                                        transform= transforms.ToTensor(),
                                        download=True)

test_dataset = datasets.MNIST(root='dataset/mnist',
                                    train=False,
                                    transform= transforms.ToTensor(),
                                    download=True)

train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=32,
                                    shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=32,
                                    shuffle=False)


class simapleNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(4*4*20,10,True)
        self.relue = torch.nn.ReLU()

    def forward(self,x):
        
        batch_size = x.size(0)
        x=self.relue(self.maxpool(self.conv1(x)))
        x=self.relue(self.maxpool(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=self.fc(x)
        
        return x
        
    

model = simapleNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    runing_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        runing_loss += loss.item()

        if batch_idx % 300 ==299:
            print(f'{epoch+1},{batch_idx+1:5d} loss : {runing_loss/1000}')
            runing_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target =data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(target)
            _,predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            corect +=(predicted==target).sum.item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
     

if __name__ == "__main__":
    for i in range(10):
        train(i)
    
    test()







