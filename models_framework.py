import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_EPOCHS = 1
LR = 1e-3
NUM_CLASSES = 9

DATA_PATH = "/home/snowwy/data/train"

# =========================
# DATA
# =========================

def get_loaders(train_transform):
    full_dataset = datasets.ImageFolder(DATA_PATH, transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


baseline_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

vit_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# =========================
# MODELS
# =========================

def get_resnet():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def get_mobilenet():
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    return model.to(device)


def get_vgg():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    return model.to(device)


def get_alexnet():
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    return model.to(device)


def get_efficientnet():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(device)


def get_vit_light(freeze=True):
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True

    return model.to(device)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 56 * 56, NUM_CLASSES)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')

    return acc, f1


def train_model(model, train_loader, val_loader, name="model"):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc, f1 = evaluate(model, val_loader)

        print(f"[{name}] Epoch {epoch+1}")
        print(f"Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{name}_best.pth")

    return best_f1

# =========================
# EXPERIMENTS
# =========================

results = {}

# 🔹 Baseline
train_loader, val_loader = get_loaders(baseline_transforms)
results["resnet_baseline"] = train_model(get_resnet(), train_loader, val_loader, "resnet_baseline")

# 🔹 Improved
train_loader, val_loader = get_loaders(train_transforms)
results["resnet_improved"] = train_model(get_resnet(), train_loader, val_loader, "resnet_improved")

# 🔹 CNN models comparison
train_loader, val_loader = get_loaders(train_transforms)

results["resnet"] = train_model(get_resnet(), train_loader, val_loader, "resnet")
results["mobilenet"] = train_model(get_mobilenet(), train_loader, val_loader, "mobilenet")
results["efficientnet"] = train_model(get_efficientnet(), train_loader, val_loader, "efficientnet")
results["vgg"] = train_model(get_vgg(), train_loader, val_loader, "vgg")
results["alexnet"] = train_model(get_alexnet(), train_loader, val_loader, "alexnet")
results["cnn"] = train_model(SimpleCNN().to(device), train_loader, val_loader, "cnn")

# 🔹 Transformer
train_loader, val_loader = get_loaders(vit_transforms)
results["vit_light"] = train_model(get_vit_light(), train_loader, val_loader, "vit_light")

# =========================
# RESULTS
# =========================

print("\n=== FINAL RESULTS ===")
for k, v in results.items():
    print(f"{k}: F1 = {v:.4f}")