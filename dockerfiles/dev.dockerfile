FROM train:latest AS base

# Install additional development dependencies
RUN pip install -r requirements_dev.txt --no-cache-dir --verbose

# Set working directory for development
WORKDIR /workspace

# Ensure bash is the default shell for dev container
CMD ["bash"]
