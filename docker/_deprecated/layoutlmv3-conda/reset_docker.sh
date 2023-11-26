#!/bin/bash

echo "Stopping docker"
sudo systemctl stop docker

sudo systemctl stop docker.socket

echo "Checking if docker is running"
docker ps -a
sleep 1

echo "Pausing for 5 seconds"
sleep 5

echo "Restarting docker"
sudo systemctl start docker

docker ps -a