import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === Step 1: Preprocessing ===
print("--> Step 1: Preprocessing Images")
DATA_DIR = 'coins_dataset'
classes = ['1rs', '2rs', '5_1rs', '5_2rs', '10rs']
IMAGE_SIZE = (128, 128)

processed_images = []
labels = []
# Converts class names to integer labels, e.g., '1rs' → 0, '2rs' → 1
label_map = {cls: idx for idx, cls in enumerate(classes)}

# Each image is flattened to a 1D vector of length 128 × 128 = 16384
for cls in classes:
    cls_folder = os.path.join(DATA_DIR, cls)
    for img_name in tqdm(os.listdir(cls_folder), desc=f"Processing {cls}"):
        img_path = os.path.join(cls_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray / 255.0
        processed_images.append(img_normalized.flatten())
        labels.append(label_map[cls])

X = np.array(processed_images)
y = np.array(labels)

X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

print(f"Processed dataset shape: {X.shape}, Labels shape: {y.shape}")

# === Step 2: One-hot Encoding ===
print("\n--> Step 2: One-hot Encoding")
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

Y = one_hot_encode(y, len(classes))

# === Step 3: Train-test Split ===
print("\n--> Step 3: Splitting Dataset")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=y)

# === Step 4: Activation Functions and Utilities ===
print("\n--> Step 4: Activation Functions and Utilities")
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def compute_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    return -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m

def accuracy(Y_pred, Y_true):
    return np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_true, axis=1))

# === Step 5: Neural Network Core ===
print("\n--> Step 5: Neural Network Core")
def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    return {
        'W1': np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size),
        'b1': np.zeros((1, hidden1_size)),
        'W2': np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2. / hidden1_size),
        'b2': np.zeros((1, hidden2_size)),
        'W3': np.random.randn(hidden2_size, output_size) * np.sqrt(2. / hidden2_size),
        'b3': np.zeros((1, output_size))
    }

def forward_pass(X, params, dropout_rate=0.0, train=True):
    Z1 = X @ params['W1'] + params['b1']
    A1 = relu(Z1)
    D1 = None
    if train and dropout_rate > 0:
        D1 = (np.random.rand(*A1.shape) > dropout_rate).astype(float)
        A1 *= D1
        A1 /= (1 - dropout_rate)

    Z2 = A1 @ params['W2'] + params['b2']
    A2 = relu(Z2)
    D2 = None
    if train and dropout_rate > 0:
        D2 = (np.random.rand(*A2.shape) > dropout_rate).astype(float)
        A2 *= D2
        A2 /= (1 - dropout_rate)

    Z3 = A2 @ params['W3'] + params['b3']
    A3 = softmax(Z3)

    return Z1, A1, D1, Z2, A2, D2, Z3, A3

def backward_pass(X, Y, Z1, A1, D1, Z2, A2, D2, A3, params, l2_lambda, dropout_rate):
    m = X.shape[0]
    dZ3 = A3 - Y
    dW3 = A2.T @ dZ3 / m + l2_lambda * params['W3']
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3 @ params['W3'].T
    if D2 is not None:
        dA2 *= D2
        dA2 /= (1 - dropout_rate)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2 / m + l2_lambda * params['W2']
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ params['W2'].T
    if D1 is not None:
        dA1 *= D1
        dA1 /= (1 - dropout_rate)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m + l2_lambda * params['W1']
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

# gradient descent updates
def update_parameters(params, grads, lr):
    for key in params:
        params[key] -= lr * grads['d' + key]
    return params

def train(X, Y, input_size, hidden1_size, hidden2_size, output_size,
          epochs=300, lr=0.01, batch_size=32, l2_lambda=0.001, dropout_rate=0.3, patience=15):

    print("\n   --> Training the Neural Network")
    params = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)
    best_acc = 0
    best_params = None
    patience_counter = 0

    losses = []
    accuracies = []

    for epoch in range(epochs):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            Z1, A1, D1, Z2, A2, D2, Z3, A3 = forward_pass(X_batch, params, dropout_rate, train=True)
            grads = backward_pass(X_batch, Y_batch, Z1, A1, D1, Z2, A2, D2, A3, params, l2_lambda, dropout_rate)
            params = update_parameters(params, grads, lr)

        _, _, _, _, _, _, _, train_preds = forward_pass(X, params, dropout_rate, train=False)
        train_loss = compute_loss(train_preds, Y)
        train_acc = accuracy(train_preds, Y)

        losses.append(train_loss)
        accuracies.append(train_acc)

        print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

        if train_acc > best_acc + 1e-4:
            best_acc = train_acc
            best_params = {k: v.copy() for k, v in params.items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Accuracy: {best_acc:.4f}")
            break

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_params if best_params else params

# === Step 6: Training the Model ===
print("\n--> Step 6: Training the Model")
input_size = X_train.shape[1]
hidden1_size = 512
hidden2_size = 256
output_size = len(classes)

params = train(X_train, Y_train, input_size, hidden1_size=256, hidden2_size=128, output_size=len(classes), epochs=300, lr=0.01, batch_size=32, l2_lambda=0.01, dropout_rate=0.5, patience=15)

# === Step 7: Evaluate on Test Set ===
print("\n--> Step 7: Evaluating on Test Set")
_, _, _, _, _, _, _, A3_test = forward_pass(X_test, params, train=False)
test_acc = accuracy(A3_test, Y_test) * 100
print(f"Final Test Accuracy: {test_acc:.2f}%")