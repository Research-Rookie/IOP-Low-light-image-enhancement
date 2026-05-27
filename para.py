import torch
import torch.nn as nn
from thop import profile, clever_format

from model import Network


class SCIWrapper(nn.Module):
    """
    Network.forward() returns:
        ilist, rlist, inlist, attlist

    rlist[-1] is usually used as the final enhanced output.
    This wrapper makes the model return a single tensor for THOP.
    """
    def __init__(self, model):
        super(SCIWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        ilist, rlist, inlist, attlist = self.model(x)
        return rlist[-1]


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = Network(stage=3).to(device)
    model.eval()

    wrapped_model = SCIWrapper(model).to(device)
    wrapped_model.eval()


    input_h = 256
    input_w = 256
    x = torch.randn(1, 3, input_h, input_w).to(device)

    total_params, trainable_params = count_params(model)

    print("=" * 70)
    print(f"Input size: 1 x 3 x {input_h} x {input_w}")
    print(f"Total Params: {total_params / 1e6:.6f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.6f} M")


    with torch.no_grad():
        flops, params = profile(wrapped_model, inputs=(x,), verbose=False)

    flops_fmt, params_fmt = clever_format([flops, params], "%.3f")

    print("-" * 70)
    print(f"THOP Params: {params_fmt}")
    print(f"THOP FLOPs/MACs: {flops_fmt}")
    print(f"Params (M): {params / 1e6:.6f}")
    print(f"GFLOPs/GMacs: {flops / 1e9:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()