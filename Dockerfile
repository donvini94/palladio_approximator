
FROM	nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
WORKDIR	/app
ENV	DEBIAN_FRONTEND	noninteractive
ENV	PYTHONUNBUFFERED	1
ENV	PATH	"/usr/local/cuda/bin:${PATH}"
ENV	LD_LIBRARY_PATH	"/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN	apt-get update && apt-get install -y \
	python3 \
	python3-pip \
	python3-dev \
	git \
	wget \
	&& rm -rf /var/lib/apt/lists/*
RUN	pip3 install --no-cache-dir --upgrade pip setuptools wheel
COPY	requirements.txt	/app/
RUN	pip3 install --no-cache-dir -r requirements.txt
COPY	.	/app/
ENV	NVIDIA_VISIBLE_DEVICES	all
ENV	NVIDIA_DRIVER_CAPABILITIES	compute,utility
CMD	["bash"]
