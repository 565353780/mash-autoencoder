import torch
from tqdm import trange
from mamba_ssm import Mamba, Mamba2

def test():
    batch, length, dim, headdim = 256, 400, 16, 2

    x = torch.randn(batch, length, dim).to("cuda")

    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=64,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")

    total_params = sum(p.numel() for p in model.parameters())
    print('mamba total_params :', total_params)

    for _ in trange(1000):
        y = model(x)

    assert y.shape == x.shape

    print(y.shape)

    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=64,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        headdim=headdim,
    ).to("cuda")

    total_params = sum(p.numel() for p in model.parameters())
    print('mamba2 total_params :', total_params)

    for _ in trange(1000):
        y = model(x)

    assert y.shape == x.shape

    print(y.shape)

    return True
