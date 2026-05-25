# train_lipsync_model.py
import os, json, requests, zipfile, numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============ СКАЧИВАНИЕ ДАТАСЕТА ============
def download_ravdess(data_dir="./training_data"):
    """Скачивает RAVDESS датасет."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = data_dir / "ravdess_audio.zip"
    
    if not zip_path.exists():
        print("Скачивание RAVDESS (1.5 GB)...")
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(8192), total=int(r.headers.get('content-length',0))//8192, unit='KB'):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir / "ravdess")
    return data_dir / "ravdess"

# ============ ОБРАБОТКА ДАННЫХ ============
def process_audio(audio_path):
    """Извлекает MFCC и генерирует синтетические landmarks."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        energy = librosa.feature.rms(y=y)[0]
        n_frames = mfcc.shape[1]
        
        # Генерируем landmarks (только рот двигается)
        landmarks = np.zeros((n_frames, 136))
        for frame in range(n_frames):
            mo = energy[min(frame, len(energy)-1)] * 2
            for i in range(48, 68):
                landmarks[frame, i*2] = 0.5
                landmarks[frame, i*2+1] = 0.5 + mo * 0.3
        
        return {
            'audio_features': mfcc.T.tolist(),
            'visual_features': landmarks.tolist(),
            'energy': energy.tolist(),
            'file': str(audio_path)
        }
    except:
        return None

def prepare_dataset(audio_dir):
    """Готовит датасет из аудиофайлов."""
    files = list(Path(audio_dir).rglob("*.wav"))
    print(f"Обработка {len(files)} файлов...")
    data = [process_audio(str(f)) for f in tqdm(files) if process_audio(str(f))]
    return data

# ============ МОДЕЛЬ ============
class LipSyncModel(nn.Module):
    def __init__(self, in_dim=13, hid=256, out_dim=136):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(), nn.Dropout(.2), nn.Linear(hid,hid), nn.ReLU())
        self.lstm = nn.LSTM(hid, hid, 2, batch_first=True, dropout=.2, bidirectional=True)
        self.dec = nn.Sequential(nn.Linear(hid*2,512), nn.ReLU(), nn.Dropout(.3), nn.Linear(512,256), nn.ReLU(), nn.Linear(256,out_dim), nn.Tanh())
    def forward(self, x):
        x = self.enc(x); x, _ = self.lstm(x)
        return self.dec(x.mean(1))

# ============ ДАТАСЕТ ============
class LipDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        a = torch.FloatTensor(s['audio_features'])
        v = torch.FloatTensor(s['visual_features'])
        return a, v.mean(0) if len(v.shape)>1 else v

def collate(batch):
    audio, visual = zip(*batch)
    max_len = max(a.shape[0] for a in audio)
    dim = audio[0].shape[1]
    padded = [torch.cat([a, torch.zeros(max_len-a.shape[0],dim)]) if a.shape[0]<max_len else a for a in audio]
    return torch.stack(padded), torch.stack(visual)

# ============ ОБУЧЕНИЕ ============
def train():
    print("="*60 + "\nОБУЧЕНИЕ LIP SYNC МОДЕЛИ\n" + "="*60)
    
    # 1. Данные
    path = download_ravdess()
    dataset = prepare_dataset(str(path))
    if not dataset: return print("❌ Нет данных!")
    print(f"✅ {len(dataset)} образцов")
    
    # 2. Train/val
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(LipDataset(train_data), batch_size=8, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(LipDataset(val_data), batch_size=8, shuffle=False, collate_fn=collate, drop_last=True)
    
    # 3. Модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LipSyncModel().to(device)
    print(f"Устройство: {device}, Батчей: {len(train_loader)}")
    
    # 4. Обучение
    opt = optim.Adam(model.parameters(), lr=0.001)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_loss = float('inf')
    
    for epoch in range(30):
        model.train()
        tl = sum(crit(model(a.to(device)), v.to(device)).item() for a, v in tqdm(train_loader, desc=f"Epoch {epoch+1}")) / len(train_loader)
        train_losses.append(tl)
        
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(a.to(device)), v.to(device)).item() for a, v in val_loader) / len(val_loader)
        val_losses.append(vl)
        sch.step(vl)
        
        if vl < best_loss:
            best_loss = vl
            torch.save({'model_state_dict': model.state_dict()}, 'best_lipsync_model.pth')
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train={tl:.4f}, Val={vl:.4f}")
    
    # 5. График
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, 'b-', label='Train', lw=2)
    plt.plot(val_losses, 'r-', label='Val', lw=2)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Кривые обучения'); plt.legend(); plt.grid(alpha=.3)
    plt.savefig('training_curves.png', dpi=150); plt.show()
    
    # 6. Сохранение
    torch.save({'model_state_dict': model.state_dict()}, 'final_lipsync_model.pth')
    json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'best_loss': best_loss}, open('training_results.json','w'), indent=2)
    
    print(f"✅ Готово! Лучшая loss: {best_loss:.4f}")
    return model

if __name__ == "__main__":
    train()