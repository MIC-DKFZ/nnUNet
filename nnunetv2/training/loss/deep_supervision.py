from torch import nn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any(x != 0 for x in weight_factors), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all(isinstance(i, (tuple, list)) for i in args), (
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        )
        return sum(w * self.loss(*inputs) for w, *inputs in zip(self.weight_factors, *args, strict=True) if w != 0.0)
