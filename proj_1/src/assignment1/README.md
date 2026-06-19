# DDPM Project: Training and Evaluation (MNIST)

This repository contains the implementation of a **Denoising Diffusion
Probabilistic Model (DDPM)** trained on the MNIST dataset. You will use this
code to explore training dynamics, architecture, and the reverse diffusion
process as part of your homework assignment.

## 🛠 Setup Instructions

The dependencies for this project are managed via `pyproject.toml`. Ensure you have Python 3.9+ installed.

1. **Clone the repository:**
```bash
git clone https://github.com/taariqnazar/diffusion-models.git
cd diffusion-models

```


2. **Install dependencies:**
```bash
pip install .

```


*Key dependencies include `torch >= 2.8.0`, `torchvision`, `tensorboard`, and `tqdm`.*

---

## 📂 Project Structure & Homework Mapping

The code is organized to help you locate specific components required for your report:

| Task | Focus Area | Relevant Files |
| --- | --- | --- |
| **Task 2** | Time Embedding | `src/assignment1/model.py` (See `TimeEmbedding` and `ResidualBlock`) |
| **Task 3/4** | Training Dynamics | `src/assignment1/train.py` |
| **Task 5** | Reverse Process | `src/assignment1/sample.py` |
| **Task 6** | FID & IS | `src/assignment1/classifier/` |

---

## Running the Code

### 1. Training the DDPM (Tasks 3 & 4)

To start training the diffusion model on MNIST, run:

```bash
python src/assignment1/train.py

```

* 
**Epochs:** The model is set to train for **50 epochs**.


* 
**Checkpoints:** The script saves checkpoints every 5 epochs in the `checkpoints/` directory.


* **Visualization:** Training/validation loss and generated samples are logged to **TensorBoard**. To view them, run:
```bash
tensorboard --logdir runs/

```



### 2. The Reverse Process (Task 5)

The `sample.py` script contains the logic for the reverse diffusion process. To generate the "denoising timeline" required for Task 5, you will need to modify the loop in `sample.py` to save the intermediate  predictions at the specific timesteps .

### 3. Quantitative Evaluation (Task 6)

Before calculating FID and IS, you must have a trained classifier.

1. **Train the Classifier:**
```bash
python src/assignment1/classifier/train.py

```


2. **Evaluation:** Use the `load_classifier` function in
   `src/assignment1/classifier/helper.py` to load your trained ResNet model and
evaluate your generated DDPM samples.
---

Here you will need to implement the logic to compute FID and IS scores for your
generated samples. The `src/assignment1/classifier/` directory contains helper
functions and scripts to assist you in this evaluation.

## Implementation Details for Your Report

### Positional Encoding (Task 2)

In `model.py`, look for the `TimeEmbedding` class. Note how the scalar  is
transformed into a vector of size `dim` using sinusoidal functions before being
injected into the `ResidualBlock` via a Linear layer and SiLU activation.

### The Reverse Step

The sampling logic in `sample.py` implements the following reverse step formula
from Ho et al.:

Where  is your UNet's noise prediction.

