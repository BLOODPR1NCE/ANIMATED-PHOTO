# train_lipsync_model.py
import os
import json
import requests
import zipfile
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LipSyncDatasetDownloader:
    """Скачивание бесплатных датасетов без регистрации."""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_ravdess_audio(self):
        """RAVDESS. Аудио с эмоциями + можно извлечь визуальные признаки."""
        print("="*60)
        print("СКАЧИВАНИЕ RAVDESS DATASET")
        print("="*60)
        
        url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        zip_path = self.data_dir / "ravdess_audio.zip"
        
        if not zip_path.exists():
            print("Скачивание RAVDESS...")
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("Распаковка...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir / "ravdess")
        
        return self.data_dir / "ravdess"


class LipSyncDataProcessor:
    """Обработка аудио для обучения lip sync модели."""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def extract_features(self, audio_path: str) -> dict:
        """Извлекает аудио признаки и генерирует визуальные ключевые точки."""
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=3.0) # Загрузка аудио
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCC признаки (как движение губ)
            energy = librosa.feature.rms(y=y)[0] # Энергия (амплитуда открытия рта)
            zcr = librosa.feature.zero_crossing_rate(y)[0] # Zero-crossing rate (артикуляция)
            n_frames = mfcc.shape[1] # Генерируем синтетические ключевые точки лица
            landmarks = np.zeros((n_frames, 136)) # (68 точек × 2 координаты = 136 значений)
            
            # Моделируем движение губ на основе аудио
            for frame in range(n_frames):
                mouth_open = energy[min(frame, len(energy)-1)] * 2 # Энергия определяет открытие рта
                
                # Губы (точки 48-67) - основные для lip sync
                for i in range(48, 68):
                    idx = i * 2
                    landmarks[frame, idx] = 0.5  # x координата
                    landmarks[frame, idx+1] = 0.5 + mouth_open * 0.3  # y координата
            
            return {
                'mfcc': mfcc,
                'energy': energy,
                'zcr': zcr,
                'landmarks': landmarks,
                'audio': y
            }
        except Exception as e:
            print(f"Ошибка обработки {audio_path}: {e}")
            return None
    
    def prepare_dataset(self, audio_dir: str) -> list:
        """Подготавливает датасет для обучения."""
        audio_files = list(Path(audio_dir).rglob("*.wav"))
        dataset = []
        
        print(f"Обработка {len(audio_files)} аудиофайлов...")
        
        for audio_path in tqdm(audio_files):  # Ограничим для скорости
            features = self.extract_features(str(audio_path))
            
            if features:
                dataset.append({
                    'audio_features': features['mfcc'].T.tolist(),  # (время, 13)
                    'energy': features['energy'].tolist(),
                    'visual_features': features['landmarks'].tolist(),  # (время, 136)
                    'file': str(audio_path)
                })
        
        return dataset


class LipSyncModel(nn.Module):
    """Нейросеть для синхронизации губ."""
    def __init__(self, input_dim=13, hidden_dim=256, output_dim=136):
        super().__init__()
        
        # Аудио энкодер
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM для временных зависимостей
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Декодер в ключевые точки лица
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Нормализованные координаты
        )
        
    def forward(self, audio_features):
        # audio_features: (batch, time, 13)
        batch_size, seq_len, _ = audio_features.shape
        
        # Кодируем аудио
        x = self.audio_encoder(audio_features)  # (batch, time, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden_dim*2)
        
        # Декодируем в ключевые точки
        landmarks = self.decoder(lstm_out)  # (batch, time, 136)
        
        # Усредняем по времени для одного кадра
        landmarks_mean = landmarks.mean(dim=1)  # (batch, 136)
        
        return landmarks_mean


class LipSyncDataset(Dataset):
    """Датасет для обучения."""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        audio = torch.FloatTensor(sample['audio_features']) # Аудио признаки
        visual = torch.FloatTensor(sample['visual_features']) # Визуальные признаки (ключевые точки)
        visual_mean = visual.mean(dim=0) if len(visual.shape) > 1 else visual # Усредняем визуальные признаки по времени
        
        return audio, visual_mean


def collate_fn(batch):
    """Паддинг для разной длины аудио."""
    audio_list, visual_list = zip(*batch)
    
    # Находим максимальную длину
    max_len = max(a.shape[0] for a in audio_list)
    feature_dim = audio_list[0].shape[1]
    
    # Паддинг аудио
    padded_audio = []
    for a in audio_list:
        if a.shape[0] < max_len:
            pad = torch.zeros(max_len - a.shape[0], feature_dim)
            a = torch.cat([a, pad], dim=0)
        padded_audio.append(a)
    
    return torch.stack(padded_audio), torch.stack(visual_list)


def train_model():
    """Полный пайплайн: скачивание → подготовка → обучение."""
    print("="*60)
    print("ОБУЧЕНИЕ LIP SYNC МОДЕЛИ")
    print("="*60)
    
    # 1. Скачивание датасета
    downloader = LipSyncDatasetDownloader()
    ravdess_path = downloader.download_ravdess_audio()
    
    # 2. Подготовка данных
    processor = LipSyncDataProcessor()
    dataset = processor.prepare_dataset(str(ravdess_path))
    
    if len(dataset) == 0:
        print("❌ Нет данных для обучения")
        return
    
    print(f"\n✅ Подготовлено {len(dataset)} образцов")
    
    # Сохраняем обработанные данные
    processed_path = processor.processed_dir / "processed_data.json"
    with open(processed_path, 'w') as f:
        json.dump(dataset, f)
    
    # 3. Разделение на train/val
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_dataset = LipSyncDataset(train_data)
    val_dataset = LipSyncDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # 4. Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LipSyncModel().to(device)
    
    print(f"\nУстройство: {device}")
    print(f"Обучающих батчей: {len(train_loader)}")
    print(f"Валидационных батчей: {len(val_loader)}")
    
    # 5. Обучение
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    epochs = 30
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for audio, visual in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio = audio.to(device)
            visual = visual.to(device)
            
            optimizer.zero_grad()
            output = model(audio)
            loss = criterion(output, visual)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for audio, visual in val_loader:
                audio = audio.to(device)
                visual = visual.to(device)
                
                output = model(audio)
                loss = criterion(output, visual)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_lipsync_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    fig, axes = plt.subplots(1, 1, figsize=(14, 5))
        
    axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='blue')
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='red')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Кривые обучения модели', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 График обучения сохранен: {'training_curves.png'}")
    
    # 6. Сохранение результатов
    print(f"\n✅ Обучение завершено!")
    print(f"Лучшая val loss: {best_val_loss:.4f}")
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs': epochs
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Сохраняем финальную модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 13,
            'hidden_dim': 256,
            'output_dim': 136
        }
    }, 'final_lipsync_model.pth')
    
    print("✅ Модель сохранена: final_lipsync_model.pth")
    
    return model


if __name__ == "__main__":
    trained_model = train_model()