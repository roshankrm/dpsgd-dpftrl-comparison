# CS798L_DPSGD_Roshan_Devansh

An end-to-end PyTorch implementation and comparison of Differentially Private Stochastic Gradient Descent (DP-SGD) and Differentially Private Follow-The-Regularized-Leader (DP-FTRL) on MNIST. This repository provides tree-aggregation noise, FTRL optimizer, privacy accounting, and training scripts with configurable hyperparameters.

## Overview

- **DP-SGD** via Opacus for per-sample gradient clipping and Gaussian noise  
- **DP-FTRL** optimizer with tree-aggregation noise reuse (O(log T) variance)  
- Privacy accounting: RDP → (ε, δ) conversion for Gaussian mechanisms  
- Example models: fully-connected MLP and LeNet-5 CNN on MNIST  
- Configurable hyperparameters: batch size, epochs, learning rate, clip norm, noise multiplier, momentum, target ε  
- Automatic logging of loss, accuracy, and ε over epochs  
- Plot generation for training metrics and privacy budget  

## Installation

1. Clone this repository:  
    ```
    git clone https://github.com/ghosh-sarbajit/CS798L_DPSGD_Roshan_Devansh.git
    cd CS798L_DPSGD_Roshan_Devansh
    ```

2. (Optional) Create and activate a virtual environment:
    Use python version 3.8.10
   ```
    python3 -m venv env
    source env/bin/activate
   ```

4. Install Dependencies:
    ```
    pip install -r requirements.txt
    ```

## Scripts for running the code

1. For DP-SGD using MLP: 
    ```
    cd dp_mlp_code
    python3 dp_mlp.py
    ```

3. For DP-SGD using CNN (LeNet Architecture) with fixed Noise Multiplier (sigma) (The accountant returns the final privacy budget used): 
    ```
    cd dp_lenet_code
    python3 dp_lenet.py
    ```

5. For DP-SGD using CNN (LeNet Architecture) with fixed privacy budget (epsilon) (The accountant returns the Noise Multiplier (sigma) used):
   ``` 
   cd dp_lenet_code`
   python3 dp_lenet_with_fixed_epsilon.py
   ```

7. For DP-FTRL: 
   ```
   cd dp_ftrl
   python3 main.py --dp_ftrl --epochs=10 --batch_size=500 --l2_norm_clip=1.1 --noise_multiplier=6.36  --learning_rate=1.0
   ```
