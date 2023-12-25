# DistributedDataParallel
- [ ] https://www.vldb.org/pvldb/vol13/p3005-li.pdf
- [ ] https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md
- [ ] https://pytorch.org/tutorials/advanced/generic_join.html

# Model parallelism
- [ ] https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

# Extras
- [ ] https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
- [ ] https://pytorch.org/docs/stable/distributed.optim.html
- [ ] https://pytorch.org/docs/stable/distributed.elastic.html

# FSDP
- [ ] https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- [ ] https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?highlight=fsdp
- [ ] https://pytorch.org/blog/large-scale-training-hugging-face/]
- [ ] https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f
- [ ] https://huggingface.co/docs/accelerate/usage_guides/fsdp
- [ ] https://pytorch.org/docs/stable/fsdp.html
- [ ] https://arxiv.org/pdf/2304.11277.pdf

# Doubts
- [ ] What is the use of DataParallel in a single device?
- [ ] On a single card, does it get the same set of inputs for every iteration?
- [ ] Processgroup - Gloo, NCCL, MPI
- [ ] How to enable uneven inputs across the processes?
	* example 1:
		num_data_samples = 999
		world_size = 4
		batch_size = 30
	* How do we handle, if only a few of the processes require additional iteration?
- [ ] Usage: register_comm_hook, no_sync, join_hook, join, gradient_as_bucket_view, static_graph
- [ ] torch.mp.spawn
- [ ] Combine DDP with model parallelism
- [ ] Initialize DDP with torch.distributed.run/torchrun