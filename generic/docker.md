# Docker

Docker container is a runtime environment that contains everything to run an application, allowing it to run independetly on any system. Docker is a open-sourced platform that helps to build, ship, and run applications in containers.

## Installation
Install docker engine on Ubuntu using https://docs.docker.com/engine/install/ubuntu/

Verify the installation by running `hello-world` image
```
sudo docker run hello-world
```
Add docker your user profile's groups to run docker commands
```
sudo groupadd docker
sudo usermod -aG docker $USER
```

## Stages of containerization
* Write Docker File.
  * Dockerfile contains a set of instructions to build the docker image, generally named as Dockerfile without any extension.
* Build Docker Image.
  * Docker image is a stack of multiple layers created from Dockerfile instructions.
* Run Docker Container.
  * Container is runtime instance of docker image.

```bash
# Sample Dockerfile

ARG CODE_VERSION=24.04
FROM ubuntu:${CODE_VERSION} #Get base image
RUN apt-get update && apt-get install -y curl \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/list/*
ENV SHELL /bin/bash #sets environment variable
```

| Task | Command |
| --- | --- |
| Build DockerFile and generate docker image | docker build -t <IMAGE_NAME> . | 
| List all docker images | docker images |
| Pull docker image | docker pull <IMAGE_URL> |
| Search docker image in docker hub | docker search registry |
| Delete docker image | docker image rm <IMAGE_NAME:TAG> or docker rmi <IMAGE_ID> |
| Create container instance | docker container create -itd IMAGE --name <CONTAINER_NAME> |
| Create and run container instance | docker container run -itd IMAGE --name <CONTAINER_NAME> |
| List all containers | docker ps -a |
| Start container | docker container start <CONTAINER_NAME>|
| Restart container after T sec | docker container restart --time T <CONTAINER_NAME>|
| Rename container | docker container rename <OLD_NAME> <NEW_NAME> |
| Stop container | docker container stop <CONTAINER_NAME>|
| Delete container | docker container rm <CONTAINER_NAME> |
| Delete all the stopped containers | docker container prune |
| Run a command on a running container | docker exec <CONTAINER_NAME> <COMMAND> |
