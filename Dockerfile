FROM python:3.11-bookworm

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

## Bake the TabFM weights into the image.
##
## tabfm fetches google/tabfm-1.0.0-pytorch from the Hugging Face hub on first
## use. Doing that during a scored run would put a 6 GB download inside the
## inference time limit and make the run depend on the hub being reachable from
## the evaluation host, so it happens at build time instead. HF_HOME is set
## first so the cache lands somewhere the runtime user can read.
ENV HF_HOME=/challenge/.hf_cache
RUN python -c "\
from tabfm import tabfm_v1_0_0_pytorch as hub; \
hub.load('classification', device='cpu'); \
print('TabFM weights cached')"

## The model runs on GPU when one is present and falls back to CPU otherwise.
## A GPU is requested on the submission form: in-context inference carries the
## whole training set through every forward pass, which costs about 230 s per
## batch on an A30 and does not complete in useful time on CPU.
