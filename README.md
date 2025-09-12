# Overview

This project was developed for **Image Classification** and **Object Detection** tasks.
During my experiments, I realized that rewriting all the code from scratch for each new experiment led to a lot of redundancy.
Therefore, I tried out other third-party frameworks, such as **Lightning**, **MMDetection** and **Detectron2**.
However, I found these frameworks to be too heavy for my needs, and it was also difficult to modify their source code.
After consideration, I concluded that what I really needed was a framework that is lightweight, easy to use and modified.

For the reasons mentioned above, this project was created.
It is named ==TorchSweetie==, meaning to make PyTorch "sweeter" -- aiming to bring you a more pleasant experience when using it.

# Installation

First of all, all my experiments were conducted on a remote **Linux** server.
Therefore, althought it is theoretically feasible, I cannot guarantee that the project will run properly on macOS or Windows.

Then, I used **Miniconda** and **Miniforge** to manage the Python environment.
You may use any virtual environment manager you prefer, but in the following sections I will use `conda` as an example.

## Create Virtual Environment

```bash
$ conda create -n TorchSweetie python=3.12
$ conda activate TorchSweetie
```

## Install Project

Since the project is lightweight and easy to modified, I recommend to install it in editable mode.

```bash
$ git clone https://github.com/SikaAntler/TorchSweetie.git
$ cd TorchSweetie/
$ pip install -e . --config-settings editable_mode=strict
```
