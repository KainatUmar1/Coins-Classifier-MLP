# Coins Classifier Using MLP in Python
A lightweight **Multilayer Perceptron (MLP)** that classifies coin images into five denominations. 
**Dataset:** 500 manually captured images (100 per class) · Written entirely in **NumPy** 🐍

## 📑 Table of Contents
1. Project Overview
2. Dataset
3. Features
4. Installation
5. Usage
6. Model Architecture
7. Training Details
8. Results
9. Future Work
10. Author

## 📖 Project Overview
We walk through every step of building a coin classifier from scratch:

- **🖼️ Image preprocessing** – resize, convert to grayscale, normalize

- **Flatten images** into vectors for the MLP

- **🔢 One-hot encode** class labels

- **🧠 MLP** with two hidden layers, ReLU, dropout, and L2 regularization

- **🚀 Mini-batch training** with early stopping

- **📊 Evaluate** accuracy on a held-out test set

## 📂 Dataset
Property	Value
Images	500
Classes	1rs, 2rs, 5_1rs, 5_2rs, 10rs
Images / class	100
Resolution	128 × 128 px (grayscale)

<details> <summary>Folder structure</summary>
coins_dataset/
│
├── 1rs/
├── 2rs/
├── 5_1rs/
├── 5_2rs/
└── 10rs/
</details>

## ✨ Features
- 🏷️ Manual dataset collection & preprocessing

- ⚙️ Pure-NumPy forward/backward passes

- 💧 Dropout & L2 regularization

- ⏱️ Early stopping to curb overfitting

- 📈 Real-time training curves with Matplotlib

## ⚙️ Installation
bash
git clone https://github.com/yourusername/coins-classifier.git
cd coins-classifier
pip install numpy opencv-python matplotlib tqdm scikit-learn

## 🚀 Usage
1. Place your images inside coins_dataset/ using the folder structure above.

2. Run the script:
bash
python coins_classifier.py
You’ll see preprocessing logs, epoch-by-epoch metrics, final test accuracy, and training plots.

## 🏛️ Model Architecture
Layer	Size	Activation	Extra
Input	16 384	–	128×128 grayscale flatten
Hidden 1	256	ReLU	50 % dropout
Hidden 2	128	ReLU	50 % dropout
Output	5	Softmax	–

## 🏋️ Training Details
- **Loss:** Cross-entropy

- **Optimizer:** Mini-batch gradient descent (batch = 32)

- **Learning rate:** 0.01

- **Epochs:** up to 300 (early stopping, patience = 15)

- **Regularization:** L2 (λ = 0.01)

- **Dropout:** 0.5 during training

## 📊 Results
- Achieved ≈ 𝑓𝑖𝑛𝑎𝑙𝑡𝑒𝑠𝑡𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦% accuracy on the test set 🎉

- Early stopping kicked in to prevent overfitting.

- Training loss & accuracy curves are auto-plotted for quick inspection.

## 🔮 Future Work
- 📸 Add more diverse coin images (different lighting & backgrounds).

- 🕸️ Switch to a CNN for richer feature extraction.

- 🧪 Apply data augmentation for robustness.

- 🌐 Deploy as a lightweight web or mobile app.

## Author
- **Kainat Umar** - *Developer of this `Coins_Classification_Model` using MLP.*
