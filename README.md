# oracle_mnist

A short description of the project. TEST

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


# Project Description
### Goal
This is the project description for Group 19 in the 02476 Machine Learning Operations course at DTU. The goal of this project is to create a complete pipeline, from raw hieroglyph data to classified images, using the tools provided during the course.

### Framework
The TIMM framework appears to be the optimal choice for this project. Specifically, we plan to use an image classification framework, along with models such as ResNet18, MobileNetV3, and a custom-made convolutional neural network.

### Data
We have chosen to work with a raw hieroglyph dataset consisting of 30,222 ancient labeled characters from 10 categories of images in various sizes and colors. There is also a preprocessed version of the dataset, where the images have been converted to grayscale, negated, resized, and extended.

### Models
We plan to use several pre-trained CNN models for image classification. We are likely to experiment with different models available in the TIMM framework.


# Installation

This repository uses Conda to manage the virtual environment. If you are not using Conda, you most manually install the requirements in the ``requirements.txt`` file.

To install using Conda, make sure that ``invoke`` is installed in the ``base``-environment. If not, please run:
```
pip install invoke
```

We recommend running the repository from inside a Docker container.
With ``invoke`` installed, just run
```
invoke build_train
```
to build the training image.  
To building the image for the frontend/backend model interaction:
```
invoke build_backend
```

# Logging with Weights and Biases
To set up logging with weights and biases, insert an API key in a ``.env`` file.
The file should then have the following structure
```
WANDB_API_KEY=<your-api-key>
```
It is also possible to logging through the terminal, in the case where a API key is not configured.


# MLOps Architecture

![mlachitecture](reports/figures/ml-pipeline.png)
