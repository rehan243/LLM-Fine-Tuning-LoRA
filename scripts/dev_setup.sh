#!/bin/bash

# this script sets up the development environment
# it runs linting and tests

set -e  # exit on error

# define variables
LINTER="flake8"
TEST_CMD="pytest"
DOCKER_IMAGE="my_image:latest"

# check for required tools
if ! command -v $LINTER &> /dev/null; then
    echo "$LINTER could not be found, installing..."
    pip install flake8
fi

if ! command -v $TEST_CMD &> /dev/null; then
    echo "$TEST_CMD could not be found, installing..."
    pip install pytest
fi

# lint the code
echo "running linter..."
$LINTER src/ scripts/ tests/

# run tests
echo "running tests..."
$TEST_CMD

# build docker image (optional)
if [[ "$1" == "docker" ]]; then
    echo "building docker image..."
    docker build -t $DOCKER_IMAGE .
fi

echo "dev setup complete"  # all done!