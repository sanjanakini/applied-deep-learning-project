# Deep Learning Mini Project

This project demonstrates **applied deep learning models** including CNN, RNN/LSTM/GRU, and GAN using **PyTorch**.  
It is structured to show basic implementation, training, evaluation, and visualization of results.

## Project Structure
deep-learning-mini-project/
├── cnn/
│ ├── simple_cnn.py
│ ├── transfer_model.py
│ ├── train_cnn.py
├── rnn/
│ ├── rnn_models.py
│ ├── train_rnn.py
├── gan/
│ ├── gan_model.py
│ ├── train_gan.py
├── outputs/
│ ├── plots/
│ ├── generated_images/
├── utils/
│ ├── helper.py
├── requirements.txt
└── README.md

---

## Setup Instructions

1. Clone the repository:
git clone <your_repo_url>
cd deep-learning-mini-project

2.Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

3.Install required packages:
pip install -r requirements.txt

4.Make sure outputs/plots and outputs/generated_images exist (they will be created automatically if not)

**Running the Project**
1) CNN (Image Classification)

Train and evaluate CNN / Transfer Learning models:

python cnn/train_cnn.py
Uses CIFAR-10 dataset
Includes simple CNN and ResNet18 transfer learning
Outputs:
Training loss curve (outputs/plots/resnet_loss.png)
Confusion matrix (outputs/plots/cnn_confusion_matrix.png)

2️) RNN / LSTM / GRU (Text Classification)

Train and evaluate RNN-based models:

python rnn/train_rnn.py
Uses IMDB movie review dataset (binary sentiment classification)
Compares RNN, LSTM, and GRU
Outputs:
Training loss curves (outputs/plots/RNNModel_loss.png, etc.)
Test accuracy printed in console

3️) GAN (Image Generation)

Train GAN to generate Fashion-MNIST images:

python gan/train_gan.py
Uses Generator and Discriminator from gan_model.py
Saves generated images every few epochs to outputs/generated_images/
Outputs:
Generated image samples
Training loss curves for generator and discriminator

**Utilities:**

All helper functions like plotting loss, saving images, computing accuracy, and confusion matrix are in:

utils/helper.py

Notes / Observations:
Simple CNN performs reasonably on CIFAR-10; ResNet18 transfer learning improves accuracy significantly.
LSTM and GRU converge faster and give better accuracy than vanilla RNN on IMDB dataset.
GAN generates Fashion-MNIST images gradually; can show mode collapse if training too short.
Small subsets of datasets are used for faster experimentation, can expand for full training.

Requirements:
Python 3.9+
PyTorch
torchvision
torchtext
matplotlib
seaborn
numpy

Install all dependencies with:

pip install -r requirements.txt
