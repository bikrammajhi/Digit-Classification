#!bin/bash
sudo docker build -t digits:v1 -f docker/Dockerfile .
echo "Running Docker image..."
sudo docker run -d -v "$(pwd)/models/":/digits/models/ digits:v1
echo "Completed Run ..."