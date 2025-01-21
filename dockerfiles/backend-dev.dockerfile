FROM backend:latest AS base

# Set working directory for development
WORKDIR /workspace

# Ensure bash is the default shell for dev container
CMD ["bash"]
