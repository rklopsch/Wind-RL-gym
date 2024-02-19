# Use the official NVIDIA PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:24.01-py3
# FROM pytorch/pytorch:latest

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    gfortran \
    make \
    && rm -rf /var/lib/apt/lists/*

# Clone the XCompact3D code from the Git repository
RUN git clone https://github.com/admole/Incompact3d.git 

# Set the working directory to XCompact3D code directory
WORKDIR /Incompact3d

# Compile XCompact3d code using the make command
RUN pwd \
    ls \
    git checkout my_dev \
    make

# Create a symbolic link to the compiled executable
RUN ln -s /Incompact3d/xcompact3d /usr/bin/xcompact3d

# Set the working directory to your app
WORKDIR /app

# Copy the requirements.txt file into the image
COPY requirements.txt /app/requirements.txt

# Install Python dependencies, including gpytorch
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Example: Copy your own code into the image
COPY ./ /app

# Add the current working directory to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set the working directory to your app
WORKDIR /app

# Command to run your application
CMD ["python", "./ppo/ppo.py"]

