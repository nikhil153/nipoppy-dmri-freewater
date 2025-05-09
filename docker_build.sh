#!/bin/bash

# build the docker image w/ local Dockerfile
docker build -t bcmcpher/dmri-freewater .

# add a version tag
docker tag bcmcpher/dmri-freewater bcmcpher/dmri-freewater:2.0.0

# push the image to dockerhub to pull for apptainer build
docker push bcmcpher/dmri-freewater:2.0.0

# build the apptainer version of the image
apptainer build dmri_freewater_2.0.0.sif docker://bcmcpher/dmri-freewater:2.0.0
