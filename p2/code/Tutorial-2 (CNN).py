#!/usr/bin/env python
# coding: utf-8

# # CNN based classification network for MNIST dataset

# # step 1: loading neccessary library files

# In[4]:


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


# # step 2: perform transformation on train and test data

# In[5]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
    ])


# # load train and test set

# In[6]:


batch_size = 4

train_set = torchvision.datasets.MNIST(root='./data', train =True, download =True,
                                      transform = transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 2)
test_set = torchvision.datasets.MNIST(root = './data', train =False,
                                     download = True, transform = transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size,
                                         shuffle = False, num_workers = 2)


# # step3: create the CNN network

# In[9]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # our first conv layer
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3, 3), stride = 1, padding =1)
        
        #max pool layer
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2,2 ))
        
        # our second conv layer
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3, 3), stride =1, padding =1)
        
        # our fully connected layer
        
        self.fc1 = nn.Linear(16*7*7, 16) # flatten your tensor
        
        # our second fully connected layer
        self.fc2 = nn.Linear(16, 10)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


# In[10]:


net = CNN()


# In[11]:


print(net)


# In[12]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[13]:


net = net.to(device)


# # step 4: define our loss function and optimizer

# In[14]:


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)


# # step 5: train the network

# In[16]:


train_loss = []
train_accuracy = []

for epoch in range(2):
    running_loss = 0.0
    running_acc = 0.0
    total = 0.0
    
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        
        images =images.to(device)
        labels = labels.to(device)
        
        outputs = net(images).to(device)
        
        optimizer.zero_grad()
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _,predicted = torch.max(outputs, 1)
        total += labels.size(0)
        
        running_acc += (predicted == labels).sum().item()
        running_loss += loss.item()
        
    train_loss = running_loss / total
    train_acc = 100*running_acc / total
    
    print('Epoch: [%d] loss: %.3f' % (epoch + 1, running_loss /(len(train_loader))))
    print('Epoch: ', epoch + 1, 'Training accuracy %d %%' %(train_acc))
    


# # step 6: test the network

# In[17]:


test_acc = 0.0
total = 0.0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images).to(device)
        
        _,predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        test_acc += (predicted == labels).sum().item()
        
print(f'Accuracy of the network on test images: {100 * test_acc // total}%')


# In[ ]:




