import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib

print("CNN FILE LOADED")

# ---------------- SEED (reproducibility) ----------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- DATA ----------------
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3)
    y = librosa.util.fix_length(y, size=66150)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc

def safe_extract(file_path, skipped_list):
    try:
        return extract_features(file_path)
    except Exception as e:
        skipped_list.append((file_path, str(e)))
        return None

X, y = [], []
skipped = []

real_files = [f for f in os.listdir(REAL_DIR) if f.endswith(".wav")]
fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".wav")]
print(f"Real files found: {len(real_files)}, Fake files found: {len(fake_files)}")

for i, file in enumerate(real_files):
    feat = safe_extract(os.path.join(REAL_DIR, file), skipped)
    if feat is not None:
        X.append(feat)
        y.append(0)
    if i % 20 == 0:
        print(f"  real {i}/{len(real_files)}")

for i, file in enumerate(fake_files):
    feat = safe_extract(os.path.join(FAKE_DIR, file), skipped)
    if feat is not None:
        X.append(feat)
        y.append(1)
    if i % 20 == 0:
        print(f"  fake {i}/{len(fake_files)}")

print(f"Skipped {len(skipped)} corrupt/unreadable files:")
for path, err in skipped:
    print(f"  BAD -> {path} : {err}")

X = np.array(X)
y = np.array(y)

print("Total usable samples:", len(X))
print("Class counts -> real(0):", np.sum(y == 0), " fake(1):", np.sum(y == 1))

if len(X) == 0:
    raise RuntimeError("No usable audio files found. Check dataset folders / file formats.")

# ---------------- SHAPE ----------------
X = X[:, np.newaxis, :, :]
print("X shape after reshape:", X.shape)

# ---------------- SPLIT FIRST (avoid data leakage) ----------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ---------------- NORMALIZE (using TRAIN stats only) ----------------
train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / (train_std + 1e-8)
X_val = (X_val - train_mean) / (train_std + 1e-8)
X_test = (X_test - train_mean) / (train_std + 1e-8)

# ---------------- TENSORS ----------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,
    shuffle=True
)

# ---------------- MODEL ----------------
class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()

        # ---- dynamically compute flattened size instead of hardcoding ----
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.pool(self.relu(self.bn1(self.conv1(dummy))))
            out = self.pool(self.relu(self.bn2(self.conv2(out))))
            flat_size = out.numel()

        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_shape = X_train.shape[1:]  # (1, 40, 130)
model = CNN(input_shape)

print("MODEL CREATED OK")
print("PARAM COUNT:", sum(p.numel() for p in model.parameters()))

# ---------------- TRAIN ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Optimizer created OK")

best_val_acc = 0.0
best_state = None

for epoch in range(200):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, val_pred = torch.max(val_outputs, 1)
        val_acc = (val_pred == y_val).sum().item() / len(y_val) * 100

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/200 | Loss: {total_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

print(f"Best Val Accuracy: {best_val_acc:.2f}%")

# ---------------- LOAD BEST MODEL ----------------
if best_state is not None:
    model.load_state_dict(best_state)

# ---------------- FINAL EVAL ON UNSEEN TEST SET ----------------
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test) * 100
print(f"Final Test Accuracy (best model): {accuracy:.2f}%")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "cnn_voice_model.pth")
joblib.dump({"mean": train_mean, "std": train_std}, "cnn_config.pkl")

print("MODEL SAVED SUCCESSFULLY")