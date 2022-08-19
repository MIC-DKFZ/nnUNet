verify all ranks have the same initial parameters

save debug information? 

dist.get_rank() instead of using local_rank in trainer? -> yes!

evaluate all_gather variants
https://discuss.pytorch.org/t/do-gradients-propagate-through-all-reduce-all-gather/159572
https://discuss.pytorch.org/t/will-dist-all-gather-break-the-auto-gradient-graph/47350/6
awesome_allgather_function
https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py#L4-L25