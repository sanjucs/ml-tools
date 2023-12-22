import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

class Net(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(Net, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    return x

class RandomDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_length, input_dim, output_dim):
    self.data = torch.randn(dataset_length, input_dim)
    self.labels = torch.randn(dataset_length, output_dim)

  def __getitem__(self, index):
    return {'X': self.data[index], 'y': self.labels[index]}

  def __len__(self):
    return len(self.data)

def train(rank, world_size):
  print(f'[RANK:{rank}] Setting up rank:{rank}')
  setup(rank, world_size)

  model = Net(5, 4, 2)
  model = DDP(model)

  mse_loss = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  dataset = RandomDataset(125, 5, 2)
  sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=30, sampler=sampler)

  for item in dataloader:
    optimizer.zero_grad()
    X, y = item['X'], item['y']
    # if rank == 0:
    print(f'[RANK:{rank}] input size::{X.shape}, output size::{y.shape}')
    out = model(X)
    loss = mse_loss(out, y)
    loss.backward()
    optimizer.step()

  cleanup()

def main():
  world_size = 4
  mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
  main()
