import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from logger import Logger
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

data_dir = "split_dataset"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = train_data.classes
num_classes = len(class_names)
print(f"Toplam {num_classes} sınıf bulundu.")

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            self.base.classifier[0],
            self.base.classifier[1],
            nn.Linear(self.base.classifier[2].in_features, num_classes)
        )
    def forward(self, x):
        return self.base(x)

class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_model_wts = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, num_epochs=50):
    logger = Logger(log_dir="logs_convnext_tiny", class_names=class_names)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    stopper = EarlyStopping()
    model = model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time

        logger.log_epoch(running_loss / len(train_loader), val_loss / len(val_loader), train_acc, val_acc, epoch_time)
        print(f"[{epoch+1:02d}/{num_epochs}] Acc: {train_acc:.4f} | VAcc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
        stopper(val_acc, model)
        if stopper.early_stop:
            print("Early stopping devreye girdi.")
            break
            
    print("\nEĞİTİM AŞAMASI BİTTİ.")
    logger.save_train_log()
    model.load_state_dict(stopper.best_model_wts)
    torch.save(model.state_dict(), "best_model_convnext_tiny.pt")
    print(f"   -> En iyi model 'best_model_convnext_tiny.pt' olarak kaydedildi.\n")
    print("--- Şimdi TEST aşamasına geçiliyor... Lütfen pencereyi kapatmayın. ---")
    time.sleep(3)
    return model, logger

def test_model(model, logger):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    tta_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15)])

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs = torch.zeros(images.size(0), num_classes).to(device)
            for _ in range(5):
                augmented = tta_transform(images)
                outputs = model(augmented)
                probs += F.softmax(outputs, dim=1)
            probs /= 5
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(probs.argmax(dim=1).cpu().numpy())
            y_score.extend(probs.cpu().numpy())
            
    logger.log_test(y_true, y_pred, y_score)
    print("\nTEST AŞAMASI BİTTİ.")
    logger.save_test_log()

model = Model(num_classes=num_classes)
trained_model, logger = train_model(model, num_epochs=50)
test_model(trained_model, logger)

print("\n" + "="*60)
print("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI (ConvNeXt-Tiny)")
print("Artık bu komut istemi (CMD) penceresini kapatabilirsiniz.")
print("="*60 + "\n")