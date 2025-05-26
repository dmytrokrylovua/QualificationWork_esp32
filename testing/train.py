import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from func_systrem.constants import SIGN_NAMES  # Исправлено имя модул
from classification import ImprovedCNN, training, SIZE, BATCH_SIZE, EPOCHS
import argparse
import logging
import sys
import traceback

logger = logging.getLogger(__name__)

def setup_argument_parser():
    parser = argparse.ArgumentParser(description="Навчання моделі розпізнавання дорожніх знаків")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Кількість епох (за замовчуванням: {EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Розмір батча (за замовчуванням: {BATCH_SIZE})")
    parser.add_argument("--size", type=int, default=SIZE, help=f"Розмір вхідного зображення (за замовчуванням: {SIZE})")
    parser.add_argument("--no-gpu", action="store_true", help="Вимкнути використання GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="Кількість worker-процесів для завантаження даних")
    parser.add_argument("--pin-memory", action="store_true", help="Використовувати pin_memory для прискорення передачі даних на GPU")
    return parser

def setup_gpu():
    if not torch.cuda.is_available():
        print("CUDA недоступний, буде використовуватися CPU (навчання буде повільним)")
        return torch.device("cpu")
    
    gpu_count = torch.cuda.device_count()
    print(f"Знайдено пристроїв CUDA: {gpu_count}")
    
    free_memory = -1
    best_device_id = 0
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        memory = props.total_memory / 1024**3
        
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_mem_gb = free_mem / 1024**3
        
        print(f"GPU {i}: {name}, Пам'ять: {memory:.2f} ГБ, Вільно: {free_mem_gb:.2f} ГБ")
        
        if free_mem > free_memory:
            free_memory = free_mem
            best_device_id = i
    
    torch.cuda.set_device(best_device_id)
    print(f"Обрано GPU {best_device_id}: {torch.cuda.get_device_name(best_device_id)}")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    torch.cuda.empty_cache()
    
    return torch.device(f"cuda:{best_device_id}")

def train_model(model, train_loader, val_loader, device, num_epochs=EPOCHS):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 15
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | '
                      f'Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {running_loss/(batch_idx+1):.4f} | '
                      f'Acc: {100.*correct/total:.2f}% | '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_val_acc': best_val_acc
        }
        
        torch.save(checkpoint, 'best_traffic_sign_model_checkpoint.pth')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(checkpoint, 'checkpoints/best_model_checkpoint.pth')
            torch.save(model.state_dict(), 'best_traffic_sign_model.pth')
            print(f'Збережено нову найкращу модель з точністю {val_acc:.2f}%')
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f'Раннє зупинення на епосі {epoch+1}')
            break
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return model, history

def main():
    try:
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        print(f"Параметри навчання:")
        print(f"* Епохи: {args.epochs}")
        print(f"* Розмір батча: {args.batch_size}")
        print(f"* Розмір зображення: {args.size}x{args.size}")
        print(f"* Кількість worker-процесів: {args.num_workers}")
        print(f"* Pin memory: {'Увімкнено' if args.pin_memory else 'Вимкнено'}")
        
        if args.no_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = torch.device("cpu")
            print("GPU вимкнено, використовується CPU (може бути повільно)")
        else:
            try:
                device = setup_gpu()
            except Exception as e:
                print(f"Помилка при налаштуванні GPU: {e}")
                print("Перемикаємося на CPU")
                device = torch.device("cpu")
        
        os.environ["OMP_NUM_THREADS"] = str(args.num_workers)
        os.environ["MKL_NUM_THREADS"] = str(args.num_workers)
        
        start_time = time.time()
        
        try:
            model = training(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory
            )
            
            if model is not None:
                print(f"Навчання успішно завершено!")
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Час навчання: {int(hours)} годин, {int(minutes)} хвилин, {seconds:.2f} секунд")
                
                print("Модель навчена і готова до використання")
                print("Можна запускати main.py для розпізнавання знаків")
            else:
                print("Помилка при навчанні моделі")
        except KeyboardInterrupt:
            print("\nНавчання перервано користувачем.")
            print("Остання версія моделі може бути доступна у файлі 'best_traffic_sign_model.pth'")
        except Exception as e:
            print(f"Помилка в процесі навчання: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Критична помилка: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 