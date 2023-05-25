#!/usr/bin/env bash

VERSION="v0.01"

docker build -t doduo1.umcn.nl/nnunet_for_pathology:$VERSION . && \
docker push doduo1.umcn.nl/nnunet_for_pathology:$VERSION && \

docker tag doduo1.umcn.nl/nnunet_for_pathology:$VERSION doduo1.umcn.nl/nnunet_for_pathology:latest
docker push doduo1.umcn.nl/nnunet_for_pathology:latest

