import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

def run(rank, world_size):
  print(f'[RANK:{rank}] Setting up rank:{rank}')
  setup(rank, world_size)

  group_1 = dist.new_group([0, 1])
  group_2 = dist.new_group([2, 3])

  tensor = torch.tensor([rank])

  dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group_1)
  dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT, group=group_2)
  print(f'[RANK:{rank}] data:{tensor}')

  dist.broadcast(tensor, src=3)
  print(f'[RANK:{rank}] data:{tensor}')
  cleanup()

def main():
  world_size = 4
  mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
  main()
