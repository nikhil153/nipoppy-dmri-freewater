#!/bin/bash

# build the docker image w/ local Dockerfile
docker build -t bcmcpher/nipoppy-dmri-freewater .

# add a version tag
docker tag bcmcpher/nipoppy-dmri-freewater bcmcpher/nipoppy-dmri-freewater:0.1.0

# push the image to dockerhub to pull for apptainer build
docker push bcmcpher/nipoppy-dmri-freewater:0.1.0

# build the apptainer version of the image
apptainer build nipoppy-dmri-freewater_0.1.0.sif docker://bcmcpher/nipoppy-dmri-freewater:0.1.0
