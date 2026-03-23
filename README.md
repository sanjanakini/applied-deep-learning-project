# Deep Learning Mini Project

## Overview
This project demonstrates the application of deep learning models using PyTorch, including:
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN, LSTM, GRU)
- Generative Adversarial Networks (GAN)

The project covers model implementation, training, evaluation, and visualization of results.

---

## Project Structure
deep-learning-mini-project/
│
├── cnn/
│   ├── simple_cnn.py
│   ├── transfer_model.py
│   ├── train_cnn.py
│
├── rnn/
│   ├── rnn_models.py
│   ├── train_rnn.py
│
├── gan/
│   ├── gan_model.py
│   ├── train_gan.py
│
├── outputs/
│   ├── plots/
│   ├── generated_images/
│
├── utils/
│   ├── helper.py
│
├── requirements.txt
└── README.md

---

## Setup Instructions

1. Clone the repository:
git clone <your_repo_url>
cd deep-learning-mini-project

2. Create a virtual environment (optional):
python -m venv venv

Activate environment:
Linux/macOS:
source venv/bin/activate

Windows:
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Output directories (auto-created if not present):
- outputs/plots
- outputs/generated_images

---

## Running the Project

### 1. CNN – Image Classification
python cnn/train_cnn.py

Details:
- Dataset: CIFAR-10
- Models: Simple CNN, ResNet18 (Transfer Learning)

Outputs:
- Training loss curve → outputs/plots/resnet_loss.png
- Confusion matrix → outputs/plots/cnn_confusion_matrix.png

---

### 2. RNN / LSTM / GRU – Text Classification
python rnn/train_rnn.py

Details:
- Dataset: IMDB Movie Reviews
- Task: Binary Sentiment Classification
- Models: RNN, LSTM, GRU

Outputs:
- Loss curves → outputs/plots/
- Test accuracy printed in console

---

### 3. GAN – Image Generation
python gan/train_gan.py

Details:
- Dataset: Fashion-MNIST
- Components: Generator and Discriminator

Outputs:
- Generated images → outputs/generated_images/
- Training loss curves

---

## Utilities
utils/helper.py includes:
- Loss plotting
- Accuracy calculation
- Confusion matrix
- Image saving

---

## Observations
- Simple CNN performs reasonably well on CIFAR-10
- ResNet18 improves accuracy significantly using transfer learning
- LSTM and GRU outperform vanilla RNN
- GAN improves gradually but may show mode collapse if undertrained

---

## Requirements
- Python 3.9+
- PyTorch
- torchvision
- torchtext
- matplotlib
- seaborn
- numpy

Install using:
pip install -r requirements.txt