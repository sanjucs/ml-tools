# Distributed training in PyTorch

## DataParallel
* Split mini batches of samples into multiple smaller mini batches, and run each of the smaller minibatches in parallel.
* Fundamental ops
	* replicate
	* scatter
	* gather
	* parallel_apply
* Limitations of DataParallel
	* Replicates model in every forward pass.
	* Single process - multi threaded parallelism and suffers from GIL contention.
	* Ovehead head due to scatter and gather operations.

## DistributedDataParallel
* Model replicated on every process and each process is fed with a different set of inputs. DDP takes care of gradient communication/synchronization across the processes overlaps it with gradient computation.
* Model broadcast at DDP construction time instead of every forward pass.
* Steps involved
	* Prerequisite - DDP relies on Processgroup. So the application must create a ProcessGroup before initializing DDP
	* Construction
		* Broadcast state-dict from rank:0 for all other processes.
		* Each DDP process creates its own Reducer which
			* Maps parameter gradients into buckets
			* Registers autograd hook per parameter.
	* Forward Pass - DDP takes the input, pass to the local model and run the local model.
	* Backward Pass
		* When one gradient is ready, corresponding hook will be triggered and mark as ready for reduction.
		* Once all the gradients in a buckets are ready for reduction, the Reducer will call allreduce on the bucket.
	* Optimizer step - Since allreduce synchronizes params across the processes, optimizer step is equivalent to optimizing local model.
* Paramters are never broadcasted across the processes. But buffers like batchnorm stats are broadcasted rank 0 prcoess to all other processes in every iterations.

* [example_ddp.py](/PyTorch/example_ddp.py) shows an example usage of DDP with gloo backend
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

## Doubts
- [ ] What is the use of DataPallel in a single device?
- [ ] On a single card, does it get the same set of inputs for every iteration?
- [ ] Processgroup - Gloo, NCCL, MPI
- [ ] How to enable uneven inputs across the processes?
- [ ] Usage: register_comm_hook, no_sync, join_hook, join
