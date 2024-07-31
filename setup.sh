if [ "$(uname)" == "Darwin" ]; then
  pip install open3d==0.15.1
elif [ "$(uname)" = "Linux" ]; then
  pip install -U open3d
fi

pip install -U trimesh tensorboard Cython pykdtree timm einops
# pip install -U causal-conv1d
# pip install -U mamba-ssm

# pip install -U torch torchvision torchaudio
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
