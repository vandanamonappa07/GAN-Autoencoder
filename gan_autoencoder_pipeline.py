"""
File: gan_autoencoder_pipeline.py
Author: Vandana B.S
Description:
GAN + Autoencoder + Latent Feature Classification
Converts 5 histopathology classes into 3 clinical groups:
Normal, Benign, Malignant
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DATASET PATHS
# -----------------------------
train_path = r"E:\dataset_updated\updated code\gan-autoencoder\train"
test_path  = r"E:\dataset_updated\updated code\gan-autoencoder\test"

# -----------------------------
# CLASS MAPPING (5 → 3 GROUPS)
# -----------------------------
class_mapping = {
    "Non-Tumor": 0,              # Normal
    "Viable": 1,                # Benign
    "viable non-viable": 1,     # Benign
    "Tumor": 2,                 # Malignant
    "Non-Viable-Tumor": 2       # Malignant
}

class_names = ["Normal", "Benign", "Malignant"]

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# -----------------------------
# LOAD DATASET
# -----------------------------
train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset  = datasets.ImageFolder(test_path, transform=transform)

# Convert ImageFolder class indices → class names → mapped labels
train_dataset.targets = [
    class_mapping[train_dataset.classes[label]]
    for label in train_dataset.targets
]

test_dataset.targets = [
    class_mapping[test_dataset.classes[label]]
    for label in test_dataset.targets
]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# GENERATOR (GAN)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 3, 64, 64)
        return img


# -----------------------------
# DISCRIMINATOR
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

# -----------------------------
# AUTOENCODER
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out


autoencoder = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# -----------------------------
# TRAIN AUTOENCODER
# -----------------------------
epochs = 10

for epoch in range(epochs):
    autoencoder.train()
    for images, _ in train_loader:
        images = images.to(device)

        latent, output = autoencoder(images)
        loss = criterion(output, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(model, loader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            z, _ = model(images)
            z = z.view(z.size(0), -1)

            features.append(z.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(features), np.concatenate(labels)


X_train, y_train = extract_features(autoencoder, train_loader)
X_test, y_test   = extract_features(autoencoder, test_loader)

# -----------------------------
# CLASSIFIERS
# -----------------------------
classifiers = {
    "SVM": SVC(kernel='rbf'),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=2000)
}

# -----------------------------
# EVALUATION
# -----------------------------
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\n========================")
    print("Classifier:", name)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=class_names))
