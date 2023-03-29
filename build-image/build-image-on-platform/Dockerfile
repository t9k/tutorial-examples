# use PyTorch 2.0
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# optional, install packages for infiniband network
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y libibverbs1 librdmacm1 libibumad3 && rm -rf /var/lib/apt/lists/*

# install Python packages
RUN pip install --no-cache-dir -i https://pypi.douban.com/simple/ \
    transformers \
    datasets \
    tiktoken \
    wandb
