
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
RUN	pip3 install --upgrade pip setuptools wheel
COPY	requirements.txt	/app/
RUN	pip3 install      scikit-learn==1.3.2     joblib==1.3.2     pandas==2.1.4     numpy==1.26.3     tqdm==4.66.1     seaborn==0.13.1     matplotlib==3.8.2     mlflow==2.10.0     transformers==4.37.2     tiktoken==0.5.2 \
	&& pip3 install  torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
COPY	.	/app/
ENV	NVIDIA_VISIBLE_DEVICES	all
ENV	NVIDIA_DRIVER_CAPABILITIES	compute,utility
CMD	["bash"]
