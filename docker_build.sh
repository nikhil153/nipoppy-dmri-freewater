#!/bin/bash

# build the docker image w/ local Dockerfile
docker build -t bcmcpher/dMRI-freewater .

# add a version tag
docker tag bcmcpher/dMRI-freewater bcmcpher/dMRI-freewater:1.0.0

# push the image to dockerhub to pull for apptainer build
docker push bcmcpher/dMRI-freewater:1.0.0

# build the apptainer version of the image
apptainer build dMRI-freewater_1.0.0.sif docker://bcmcpher/dMRI-freewater:1.0.0
