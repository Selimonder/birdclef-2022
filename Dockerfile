ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV PYTHONPATH=.

RUN pip install albumentations==1.0.0 timm==0.5.4 tensorboardx pandas madgrad audiomentations librosa
RUN apt update
RUN apt install wget mc -y
# Setting the working directory
WORKDIR /workspace

RUN mkdir -p logs
RUN mkdir -p weights

# Download pretrained weights for backbones
RUN mkdir -p /root/.cache/torch/hub/checkpoints/

RUN wget -O /root/.cache/torch/hub/checkpoints/tf_efficientnetv2_l_21k-91a19ec9.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21k-91a19ec9.pth
RUN wget -O /root/.cache/torch/hub/checkpoints/nfnet_l0_ra2-45c6688d.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nfnet_l0_ra2-45c6688d.pth
RUN wget -O /root/.cache/torch/hub/checkpoints/resnet34-43635321.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth
RUN wget -O /root/.cache/torch/hub/checkpoints/tf_efficientnet_b7_ns-1dbc32de.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth
RUN wget -O /root/.cache/torch/hub/checkpoints/tf_efficientnetv2_m_21k-361418a2.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m_21k-361418a2.pth


# Copying the required codebase
COPY . /workspace

RUN ["/bin/bash"]

