```bash
conda create --name sz-rlhf python=3.10

# Install pytorch
pip3 install torch torchvision torchaudio
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

# Install openrlhf
pip install openrlhf[vllm]
```
