# Use the official NVIDIA PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN useradd -m demo-user

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    gfortran \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/demo-user

# Clone the XCompact3D code from the Git repository
RUN git clone https://github.com/admole/Incompact3d.git 

# Set the working directory to XCompact3D code directory
WORKDIR /home/demo-user/Incompact3d

# Compile XCompact3d code using the make command
RUN git checkout my_dev
RUN make &> log.make

# Create a symbolic link to the compiled executable
RUN ln -s /home/demo-user/Incompact3d/xcompact3d /usr/bin/xcompact3d

# Set the working directory to your app
WORKDIR /home/demo-user/app

# Copy the requirements.txt file into the image
COPY requirements.txt /home/demo-user/app/requirements.txt

# Install Python dependencies, including gpytorch
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Example: Copy your own code into the image
COPY --chown=demo-user:demo-user ./ /home/demo-user/app
RUN chown -R demo-user:demo-user /home/demo-user/app

# Add the current working directory to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/home/demo-user/app"

USER demo-user

# Log in to WANDB for logging
ENV WANDB_API_KEY=<API_KEY>
ENV WANDB_DIR=/home/demo-user/app/outputs

# Command to run your application
CMD ["python", "./ppo/ppo.py"]

