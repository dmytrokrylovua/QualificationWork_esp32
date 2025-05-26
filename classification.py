import albumentations as A
import csv
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, Dict
from func_systrem.constants import (
    SIZE, CLASS_NUMBER, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, NUM_WORKERS, SHAPE_CLASSES, 
    COLOR_CLASSES, SIGN_NAMES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    logger.info(f"CUDA доступний. Виявлено {torch.cuda.device_count()} GPU")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    logger.warning("CUDA недоступний! Буде використовуватися CPU (це може бути дуже повільно)")

SIZE = 256  
CLASS_NUMBER = 43
BATCH_SIZE = 16 
EPOCHS = 75
LEARNING_RATE = 0.0005  
NUM_WORKERS = 8  

SHAPE_CLASSES = {
    0: "круглий",
    1: "трикутний",
    2: "квадратний",
    3: "восьмикутний"
}

COLOR_CLASSES = {
    0: "червоний",
    1: "синій",
    2: "жовтий",
    3: "білий"
}

def get_sign_name(class_id: int) -> str:
    return SIGN_NAMES.get(class_id, f"Невідомий знак (Клас {class_id})")

class TrafficSignsDataset(Dataset):    
    def __init__(self, csv_file: str, root_dir: str, transform=None, is_test: bool = False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.is_test = is_test
        
        self.basic_transform = A.Compose([
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.annotations.iloc[idx]['Path']
        class_id = self.annotations.iloc[idx]['ClassId']
        
        full_path = str(self.root_dir.parent / img_path)
        image = cv2.imread(full_path)
        if image is None:
            raise RuntimeError(f"Не вдалося завантажити зображення: {full_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            transformed = self.basic_transform(image=image)
        else:
            transformed = self.train_transform(image=image)
        
        image = transformed['image']
        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        return image, torch.tensor(class_id, dtype=torch.long)

class ImprovedCNN(nn.Module):    
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, CLASS_NUMBER)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ))
        
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            ))
        
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            ))
        
        return nn.ModuleList(layers)
    
    def forward(self, x):
        x = self.initial(x)
        
        for layer in [self.layer1, self.layer2, self.layer3]:
            identity = x
            for i, block in enumerate(layer):
                if i == 1 and len(layer) > 1:
                    identity = block(identity)
                elif i == 0:
                    x = block(x)
                else:
                    x = block(x)
            x = F.relu(x + identity)
        
        att = self.attention(x)
        x = x * att
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def load_model(model: nn.Module, model_path: str, device: torch.device) -> bool:
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        logger.info("Модель успішно завантажена")
        return True
        
    except Exception as e:
        logger.error(f"Помилка завантаження моделі: {e}")
        return False

def save_model(model: nn.Module, optimizer: optim.Optimizer, scheduler, epoch: int, 
               val_acc: float, filename: str) -> None:
    torch.save(model.state_dict(), filename)
    
    full_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
    }
    checkpoint_path = filename.replace('.pth', '_checkpoint.pth')
    torch.save(full_state, checkpoint_path)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, num_epochs: int = EPOCHS) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f"{running_loss/(progress_bar.n+1):.3f}",
                'acc': f"{100.*correct/total:.1f}%"
            })
            
            if device.type == 'cuda' and progress_bar.n % 10 == 0:
                torch.cuda.empty_cache()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        logger.info(f'Epoch {epoch+1}: Val Acc = {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_traffic_sign_model.pth')
            logger.info(f'Збережена нова найкраща модель з точністю {val_acc:.2f}%')
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            logger.info(f'Рання зупинка на епосі {epoch+1}')
            break
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def predict(model: nn.Module, image: np.ndarray, device: torch.device) -> Tuple[int, float, str]:
    model.eval()
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transformed = transform(image=image)
    image = transformed['image']
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(image)
        else:
            outputs = model(image)
        
        probabilities = F.softmax(outputs, dim=1)
        
        confidence, predicted = probabilities.max(1)
        
        predicted_class = predicted.item()
        confidence_percent = confidence.item() * 100
        
        sign_name = get_sign_name(predicted_class)
        
    if device.type == 'cuda' and torch.cuda.is_available():
        del image, outputs, probabilities, confidence, predicted
        torch.cuda.empty_cache()
        
    return predicted_class, confidence_percent, sign_name

def training(batch_size=None, num_workers=None, pin_memory=True) -> Optional[ImprovedCNN]:
    try:
        logger.info('Ініціалізація датасетів...')
        
        actual_batch_size = batch_size or BATCH_SIZE
        actual_num_workers = num_workers or NUM_WORKERS
        
        try:
            train_dataset = TrafficSignsDataset(
                csv_file='dataset/Train.csv',
                root_dir='dataset/Train',
                is_test=False
            )
            
            test_dataset = TrafficSignsDataset(
                csv_file='dataset/Test.csv',
                root_dir='dataset/Test',
                is_test=True
            )
            
            train_size = int(0.85 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, 
                [train_size, val_size]
            )
            
        except Exception as e:
            logger.error(f"Помилка створення датасету: {e}")
            return None
        
        logger.info(f'Розмір тренувального датасету: {len(train_dataset)} зображень')
        logger.info(f'Розмір валідаційного датасету: {len(val_dataset)} зображень')
        logger.info(f'Розмір тестового датасету: {len(test_dataset)} зображень')
        
        try:
            effective_num_workers = actual_num_workers
            
            if os.name == 'nt' and effective_num_workers > 0:
                logger.warning("На Windows багатопотокове завантаження іноді викликає проблеми. Використовуємо 0 workers.")
                effective_num_workers = 0
            
            logger.info(f"Використовується workers: {effective_num_workers}, pin_memory: {pin_memory}")
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=actual_batch_size, 
                shuffle=True, 
                num_workers=effective_num_workers, 
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=effective_num_workers > 0,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=actual_batch_size, 
                shuffle=False, 
                num_workers=effective_num_workers, 
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=effective_num_workers > 0
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=actual_batch_size, 
                shuffle=False, 
                num_workers=effective_num_workers, 
                pin_memory=pin_memory and torch.cuda.is_available(),
                persistent_workers=effective_num_workers > 0
            )
        except Exception as e:
            logger.error(f"Помилка створення завантажувачів даних: {e}")
            return None
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            logger.info(f'Використовується GPU: {torch.cuda.get_device_name(0)}')
        else:
            device = torch.device('cpu')
            logger.warning('GPU недоступний, використовується CPU (навчання буде повільним)')
        
        try:
            model = ImprovedCNN()
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f'Всього параметрів у моделі: {total_params:,}')
            
            if device.type == 'cuda':
                logger.info(f'Зайнято пам\'яті GPU: {torch.cuda.memory_allocated(0) / 1024 / 1024:.1f} МБ')
                logger.info(f'Зарезервовано пам\'яті GPU: {torch.cuda.memory_reserved(0) / 1024 / 1024:.1f} МБ')
        except Exception as e:
            logger.error(f"Помилка створення моделі: {e}")
            return None
        
        logger.info('Початок навчання...')
        try:
            train_model(model, train_loader, val_loader, device)
        except Exception as e:
            logger.error(f"Помилка в процесі навчання: {e}")
            return None
        
        return model
            
    except Exception as e:
        logger.error(f"Помилка при навчанні моделі: {e}")
        return None

def show_image(image, title='Image'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    try:
        model = ImprovedCNN()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        if not load_model(model, 'best_traffic_sign_model.pth', device):
            logger.error("Не вдалося завантажити модель. Запустіть спочатку train.py для навчання моделі")
            return
            
        logger.info("Модель CNN успішно завантажена")
        model.eval()
        
    except Exception as e:
        logger.error(f"Помилка ініціалізації моделі: {e}")
        return

if __name__ == '__main__':
    main()