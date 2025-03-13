import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

num_epochs = 1
batch_size = 1000

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
    self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = torch.nn.Dropout(0.25)
    self.dropout2 = torch.nn.Dropout(0.5)
    self.fc1 = torch.nn.Linear(9216, 128)
    self.fc2 = torch.nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.conv2(x)
    x = torch.relu(x)
    x = torch.nn.functional.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = torch.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = torch.nn.functional.log_softmax(x, dim=1)
    return output

def train_one_epoch(model, dist, dataloader, optimizer):
  model.train()
  total_loss = 0

  for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.nll_loss(output, target, reduction='sum')
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  
  dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)

def cleanup():
  dist.destroy_process_group()

def train(rank, world_size):
  print(f'[RANK:{rank}] Setting up rank:{rank}')
  setup(rank, world_size)

  dataset = torchvision.datasets.MNIST('./MNIST_DATA', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
  sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
  model = Net()
  model = FSDP(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(1, num_epochs+1):
    sampler.set_epoch(epoch)  
    loss = train_one_epoch(model, dist, dataloader, optimizer)
    print(f'[RANK:{rank}] epoch::{epoch}/{num_epochs} loss::{loss}')
    for idx, param in enumerate(model.parameters()):
      if idx == 0: print(f'[RANK:{rank}] {torch.sum(param)} {torch.sum(param.grad)}')
    
  cleanup()

if __name__ == '__main__':
  torch.manual_seed(0)

  world_size = 4
  mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)