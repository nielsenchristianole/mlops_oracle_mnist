import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "oracle_mnist"
PYTHON_VERSION = "3.11"
CWD = os.getcwd()


# docker commands
@task
def build_train(ctx: Context, progress: str = "plain") -> None:
    """Build docker image for training"""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def build_backend(ctx: Context, progress: str = "plain") -> None:
    """Build docker image for backend"""
    ctx.run(
        f"docker build -t backend:latest . -f dockerfiles/backend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def build_frontend(ctx: Context, progress: str = "plain") -> None:
    """Build docker image for frontend"""

    ctx.run(
        f"docker build -t frontend:latest . -f dockerfiles/frontend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train_docker(ctx: Context, no_gpu: bool = False, share_data: bool = False) -> None:
    """Run training docker container."""

    command = [
        "docker",
        "run",
        "--rm",
        f"--mount type=bind,src={CWD}/.env/,dst=/gcs/cloud_mlops_bucket/.env",
        # Mount the configs directory
        f"--mount type=bind,src={CWD}/configs/,dst=/gcs/cloud_mlops_bucket/configs",
        # Mount the lightning_logs directory
        f"--mount type=bind,src={CWD}/lightning_logs/,dst=/gcs/cloud_mlops_bucket/lightning_logs",
        # Mount the outputs directory
        f"--mount type=bind,src={CWD}/outputs/,dst=/gcs/cloud_mlops_bucket/outputs",
    ]

    if share_data:
        command.append(f"--mount type=bind,src={CWD}/data/,dst=/workspace/data")  # Mount the data directory

    if not no_gpu:
        command.append("--gpus all")  # Use GPUs

    command.append("train:latest")
    ctx.run(" ".join(command), echo=True, pty=not WINDOWS)


@task
def serve_backend(ctx: Context, model_version: int = 0) -> None:
    """Serve the backend in a docker container."""

    src = f"{CWD}/lightning_logs/version_{model_version}/checkpoints/best.onnx"
    dst = "/models/model.onnx"
    command = [
        "docker",
        "run",
        "--rm",
        "-p 6060:6060",  # Expose port 6060
        f"--mount type=bind,src={src},dst={dst}",  # Mount the model
    ]

    command.append("backend:latest")
    ctx.run(" ".join(command), echo=True, pty=not WINDOWS)


@task
def serve_frontend(ctx: Context):
    """Serve the frontend in a docker container."""
    command = ["docker", "run", "--rm", "-p 8501:8501", "frontend:latest"]  # Expose port 8501

    ctx.run(" ".join(command), echo=True, pty=not WINDOWS)


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests with coverage."""
    ctx.run(
        "coverage run --source=src -m unittest discover -s tests -p 'test_*.py'",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def sweep(ctx: Context, count: int = 3) -> None:
    """Run a WandB hyperparameter sweep."""
    ctx.run(f"python src/oracle_mnist/train_sweep.py --sweep_count {count}", echo=True, pty=not WINDOWS)
