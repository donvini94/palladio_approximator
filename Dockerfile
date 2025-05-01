
FROM	pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV	DEBIAN_FRONTEND	noninteractive
ENV	TZ	Etc/UTC
WORKDIR	/app
RUN	apt-get update && apt-get install -y \
	--no-install-recommends \
	git \
	curl \
	wget \
	build-essential \
	libboost-all-dev \
	software-properties-common \
	ca-certificates \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
ENV	PATH	/opt/conda/bin:$PATH
COPY	requirements.txt	.
RUN	pip install --no-cache-dir -r requirements.txt
RUN	pip install --no-cache-dir     scikit-learn     pandas     tqdm     numpy     joblib     transformers     mlflow     seaborn     tiktoken     matplotlib
COPY	.	.
RUN	mkdir -p data/dsl_models data/measurements models
COPY	models/torch_model.py	models/gpu_rf_model.py	models/
ENV	PYTORCH_CUDA_ALLOC_CONF	max_split_size_mb:512
COPY	docker-entrypoint.sh	/docker-entrypoint.sh
RUN	chmod +x /docker-entrypoint.sh
EXPOSE	8888
ENTRYPOINT	["/docker-entrypoint.sh"]
CMD	["/bin/bash"]
