#!/bin/bash

# this script is for development tools like linting and testing

# check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# install dependencies if not already installed
install_dependencies() {
    echo "installing dependencies..."
    pip install -r requirements.txt
}

# run linting with flake8
run_lint() {
    echo "running linting..."
    if command_exists flake8; then
        flake8 src/
    else
        echo "flake8 not found, installing..."
        pip install flake8
        flake8 src/
    fi
}

# run tests with pytest
run_tests() {
    echo "running tests..."
    if command_exists pytest; then
        pytest tests/
    else
        echo "pytest not found, installing..."
        pip install pytest
        pytest tests/
    fi
}

# main function to call when script is executed
main() {
    install_dependencies
    run_lint
    run_tests
}

# call the main function
main