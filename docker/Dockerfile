FROM nvidia/cudagl:11.4.1-runtime-ubuntu20.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get install python3.8 -y \
    && apt-get install python3-pip -y
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.8-venv \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
#     && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco

COPY ./mujoco210 /root/.mujoco/mujoco210

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# WORKDIR /mujoco_py
# # Copy over just requirements.txt at first. That way, the Docker cache doesn't
# # expire until we actually change the requirements.
# COPY ./requirements.txt /mujoco_py/
# COPY ./requirements.dev.txt /mujoco_py/

WORKDIR /hilp
# ADD . /hilp
COPY requirements.txt /hilp/requirements.txt
COPY vendor /hilp/vendor

RUN mkdir -p /root/.config/pip
RUN echo "\
[global]\n\
index-url = https://mirror.sjtu.edu.cn/pypi/web/simple\
" > /root/.config/pip/pip.conf

RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && pip3 install --upgrade flax && pip3 install patchelf
RUN pip3 install --no-cache-dir -r requirements.txt --no-deps

# Delay moving in the entire code until the very end.
RUN chmod +x /hilp/vendor/Xdummy-entrypoint
ENTRYPOINT ["/hilp/vendor/Xdummy-entrypoint"]

RUN echo "\
machine api.wandb.ai\n\
  login user\n\
  password 957d8a5cfced245b0d1816a87695d11619b38702\
" > /root/.netrc