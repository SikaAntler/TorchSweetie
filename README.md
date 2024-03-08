# Overview

This project was written when I was doing experiments on image classification and object detection tasks. I found there would be a lot of duplicated codes if I wrote each experiment from scratch. Then I experienced some third-party easy-training frameworks, like **Lightning** and **MMDetectio**. However, I found these frameworks were too heavily for me, and they were much difficult to modified. I considered what I truely required was a light tool which was easy to use and modified.

# Installation

First of all, all my experiments were doing on **Linux** server, so I could not ensure that this project runs on MacOS and Windows in normal.

Then, I am using **Miniconda** as the python environments manager, so I highly recommend you to use **Miniconda** or **Anaconda**n like me. I would use **conda** to represent both of them in following content.

## 1. Create Conda Environment

```bash
$ conda create -n TorchSweetie python=3.10
$ conda activate TorchSweetie
```

## 2. Install Project

Since the project was simple and easy to modified, I recommend to install in editable mode.

```bash
$ git clone https://github.com/SikaAntler/TorchSweetie.git
$ cd TorchSweetie/
$ pip install -e . --config-settings editable_mode=strict
```

