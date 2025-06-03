cd ..
git clone git@github.com:565353780/base-trainer.git
git clone git@github.com:565353780/point-cept.git
git clone git@github.com:565353780/mesh-graph-cut.git
git clone git@github.com:565353780/chamfer-distance.git

cd base-trainer
./dev_setup.sh

cd ../point-cept
./dev_setup.sh

cd ../mesh-graph-cut
./dev_setup.sh

cd ../chamfer-distance
./dev_setup.sh

if [ "$(uname)" == "Darwin" ]; then
  pip install open3d==0.15.1
elif [ "$(uname)" = "Linux" ]; then
  pip install -U open3d
fi

pip install -U trimesh tensorboard Cython pykdtree timm einops

# pip install -U torch torchvision torchaudio
# pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# pip install -U causal-conv1d
# pip install -U mamba-ssm
