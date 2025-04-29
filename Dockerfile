FROM	pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR	/app
RUN	apt-get update && apt-get install -y \
	git \
	curl \
	&& rm -rf /var/lib/apt/lists/*
COPY	requirements.txt	.
RUN	pip install --no-cache-dir -r requirements.txt
COPY	.	.
RUN	mkdir -p data/dsl_models data/measurements
ENV	PYTORCH_CUDA_ALLOC_CONF	max_split_size_mb:512
CMD	["/bin/bash"]
