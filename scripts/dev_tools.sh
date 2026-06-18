#!/bin/bash

# run this script to check code style and run tests

set -e  # exit on error

# define a function for linting
lint() {
    echo "linting the code..."
    flake8 src/  # check for style issues
    echo "linting completed"
}

# define a function for running tests
test() {
    echo "running tests..."
    pytest tests/  # run the test suite
    echo "all tests passed"
}

# check for docker
docker_check() {
    if ! command -v docker &> /dev/null; then
        echo "docker not found, please install it"
        exit 1
    fi
}

# define a function for building and running the docker container
docker_run() {
    echo "building docker image..."
    docker build -t lora-finetuning .  # build the docker image
    echo "running docker container..."
    docker run --rm lora-finetuning  # run the container
}

# main script logic
case "$1" in
    lint)
        lint
        ;;
    test)
        test
        ;;
    docker)
        docker_check
        docker_run
        ;;
    *)
        echo "usage: $0 {lint|test|docker}"
        exit 1
        ;;
esac

# TODO: add more commands as needed
echo "done"