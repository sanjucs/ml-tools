# Distributed training in PyTorch

## DataParallel
* Split mini batches of samples into multiple smaller mini batches, and run each of the smaller mini batches in parallel.
* Fundamental ops
	* replicate
	* scatter
	* gather
	* parallel_apply
* Limitations of DataParallel
	* Replicates model in every forward pass.
	* Single process - multi threaded parallelism and suffers from GIL contention.
	* Ovehead due to scatter and gather operations.

## DistributedDataParallel
* Replicates model on every process and each process is fed with a different set of inputs. DDP takes care of gradient communication/synchronization across the processes ane overlaps it with gradient computation.
* Model broadcast at DDP construction time instead of every forward pass.
* Steps involved
	* Prerequisite - DDP relies on Processgroup. So the application must create a ProcessGroup before initializing DDP
	* Construction
		* Broadcast state-dict from rank:0 for all other processes.
		* Each DDP process creates its own Reducer which
			* Maps parameter gradients into buckets
			* Registers autograd hook per parameter.
	* Forward pass - DDP takes the input, pass to the local model and run the local model.
	* Backward pass
		* When one gradient is ready, corresponding hook will be triggered and mark as ready for reduction.
		* Once all the gradients in a bucket are ready for reduction, the Reducer will call allreduce on the bucket.
	* Optimizer step - Since allreduce synchronizes params across the processes, optimizer step is equivalent to optimizing local model.
* Parameters are never broadcasted between the processes. But buffers like batchnorm stats are broadcasted rank 0 prcoess to all other processes in every iteration.

* [example_ddp.py](/distributed/examples/example_ddp.py) shows an example usage of DDP with gloo backend
	* setup distributed env
		* using mp.Process
		```python
		import torch.multiprocessing as mp
		processes = []
		for rank in range(world_size):
			p = mp.Process(target=train, args=(rank, world_size,))
			p.start()
			processes.append(p)

		for p in processes:
			p.join()
		```
		* using mp.spawn
		```python
		import torch.multiprocessing as mp
		mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
		```

	* Initialize distributed process group
	```python
	import torch.distributed as dist
	dist.init_process_group('gloo', rank=rank, world_size=world_size)
	```
	* Construct DDP module
	```python
	from torch.nn.parallel import DistributedDataParallel as DDP
	model = DDP(model)
	```
	* Divide inputs and outputs across the processs
	```python
	sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, shuffle=True)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
	```
	* When multiple processes are used in training, user needs to explicitily call collectives to synchronize the parameter gradients.
	```python
	# average parameter gradients across the processes
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
		param.grad.data /= dist.get_world_size()
	```
## Collectives
Collectives communicate across all processes in a group. There are total seven collectives implemented in PyTorch:
* dist.broadcast(tensor, src, group): Copies tensor from src to all other processes.
* dist.reduce(tensor, dst, op, group): Applies op to every tensor and stores the result in dst.
* dist.all_reduce(tensor, op, group): Same as reduce, but the result is stored in all processes.
* dist.scatter(tensor, scatter_list, src, group): Copies the i<sup>th</sup>  tensor in the scatter_list to the i<sup>th</sup>  process.
* dist.gather(tensor, gather_list, dst, group): Copies tensor from all processes in dst.
* dist.all_gather(tensor_list, tensor, group): Copies tensor from all processes to tensor_list on all processes.
* dist.barrier(group): Blocks all processes in group until each one has entered this function.

[example_collective.py](/distributed/examples/example_collectives.py) shows an example with PyTorch collectives

## Fully Sharded Data Parallel (FSDP)

FSDP shards model's parameters, gradients and optimizer states across the workers and optionally offload the parameters to CPUs. This results in lower memory requirements on device compared to DDP, but with an increase in communication bandwidth requirements.

* Steps involved
	* Prerequisite - Initialize prcessgroup.
	* Construction - Shard model parameters and each rank keeps its own shard.
	* Forward pass
		* Run allgather to collect all parameters from all ranks of the particular FSDP unit.
		* Takes the input on the rank, run the model.
		* Release parameter shards of other ranks.
	* Backward pass
		* Run allgather to collect all parameters from all ranks of the particular FSDP unit.
		* Run backward computation.
		* Run reduce-scatter to synchronizer gradients.
		* Release parameter shards of other ranks.
	* Optimizer step
		* Each rank optimizes it's own sharded parameters.

* case 1:
	```
	model = FSDP(model)
	```
	* There is only one FSDP instance that wraps entire model.
	* During forward and backward passes, allgather will collect full parameters, hence there won't be any memory optimization.
	* Since there is only one instance, it inherently precludes commnunication and computation overlap.


## Reference

### Distributed
* https://pytorch.org/tutorials/beginner/dist_overview.html

### DataParallel
* https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
* https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
* https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
* https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism

### DistributedDataParallel
* https://pytorch.org/docs/stable/notes/ddp.html
* https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html
* https://pytorch.org/tutorials/intermediate/dist_tuto.html
* https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

### FullyShardedDataParallel
* https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
* https://www.youtube.com/watch?v=By_O0k102PY
* https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
* https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f