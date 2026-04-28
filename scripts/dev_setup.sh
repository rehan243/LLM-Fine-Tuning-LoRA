#!/bin/bash

# this script sets up the development environment for the project

# check if python is installed
if ! command -v python3 &> /dev/null
then
    echo "python3 is not installed. please install it first."
    exit 1
fi

# create a virtual environment
echo "creating a virtual environment..."
python3 -m venv venv

# activate the virtual environment
echo "activating the virtual environment..."
source venv/bin/activate

# install required packages
echo "installing required packages..."
pip install -r requirements.txt

# run linting
echo "running linting with flake8..."
flake8 src/

# run tests
echo "running tests with pytest..."
pytest tests/

# docker option
read -p "do you want to build the docker image? (y/n) " answer
if [[ $answer == "y" ]]; then
    echo "building docker image..."
    docker build -t lora-finetuning .
    echo "docker image built successfully"
fi

echo "development setup complete. enjoy coding!" 

# TODO: add more options like cleaning up the environment or updating packages.