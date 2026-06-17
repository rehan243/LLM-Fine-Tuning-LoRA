# this target will run the application locally
run:
	# make sure to install dependencies first
	pip install -r requirements.txt

	# run the main application
	python main.py

# this target will run the tests
test:
	# run tests using pytest
	pytest tests/

# linting with flake8
lint:
	# check code quality
	flake8 src/

# docker target to build and run the container
docker:
	# build the docker image
	docker build -t llm-fine-tuning-lora .

	# run the docker container
	docker run -it --rm llm-fine-tuning-lora

# clean target to remove pyc files and cache
clean:
	# remove python cache and .pyc files
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

# help target to show available commands
help:
	@echo "Available commands:"
	@echo "  make run     - run the application locally"
	@echo "  make test    - run the tests"
	@echo "  make lint    - check code quality"
	@echo "  make docker  - build and run the docker container"
	@echo "  make clean   - remove pyc files and cache"
	@echo "  make help    - show this help message"