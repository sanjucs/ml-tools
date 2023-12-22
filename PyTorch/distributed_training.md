# Distributed training in PyTorch

## DataParallel
* Split mini batches of samples into multiple smaller mini batches, and run each of the smaller minibatches in parallel.
* Fundamental ops
	* replicate
	* scatter
	* gather
	* parallel_apply
* Issues with DataParallel
	* Replicates model in every forward pass.
	* single process - multi threaded parallelism and suffers from GIL

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

## Reference

### Distributed
* https://pytorch.org/tutorials/beginner/dist_overview.html

### DataParallel
* https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
* https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html?highlight=dataparallel#
* https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
* https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism

### DistributedDataParallel

## Doubts
- [ ] What is the use of DataPallel in a single device?
- [ ] On a single card, does it get the same set of input for every iteration?
- [ ] Processgroup

<!-- # DistributedDataParallel
- [ ] https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
- [ ] https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
- [ ] https://www.vldb.org/pvldb/vol13/p3005-li.pdf
- [ ] https://pytorch.org/docs/stable/notes/ddp.html
- [ ] https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- [ ] https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
- [ ] https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md
- [ ] https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
- [ ] https://pytorch.org/docs/stable/distributed.optim.html
- [ ] https://pytorch.org/tutorials/advanced/generic_join.html
- [ ] https://pytorch.org/docs/stable/distributed.elastic.html
- [ ] https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51

FSDP
- [https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html "https://pytorch.org/tutorials/intermediate/fsdp_tutorial.html")
- [https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?highlight=fsdp](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?highlight=fsdp "https://pytorch.org/tutorials/intermediate/fsdp_adavnced_tutorial.html?highlight=fsdp")
- [https://pytorch.org/blog/large-scale-training-hugging-face/](https://pytorch.org/blog/large-scale-training-hugging-face/ "https://pytorch.org/blog/large-scale-training-hugging-face/")
- [https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f](https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f "https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f")
- [https://huggingface.co/docs/accelerate/usage_guides/fsdp](https://huggingface.co/docs/accelerate/usage_guides/fsdp "https://huggingface.co/docs/accelerate/usage_guides/fsdp"
- [https://pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html "https://pytorch.org/docs/stable/fsdp.html")

 -->
