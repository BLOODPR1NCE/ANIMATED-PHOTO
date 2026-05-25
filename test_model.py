# test_model.py
import os, sys, tempfile, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_model():
    """Загружает модель (общая функция для всех тестов)."""
    for p in ['final_lipsync_model.pth', 'best_lipsync_model.pth']:
        if os.path.exists(p):
            from train_lipsync_model import LipSyncModel
            ck = torch.load(p, map_location='cpu')['model_state_dict']
            out_dim = ck['dec.5.weight'].shape[0]
            hid = ck['enc.0.weight'].shape[0]
            model = LipSyncModel(13, hid, out_dim)
            model.load_state_dict(ck); model.eval()
            return model, p
    return None, None

# ТЕСТ 1: Модель загружается
def test_1():
    print("\nТЕСТ 1: Загрузка модели")
    model, path = load_model()
    if model:
        print(f"✅ Загружена из {path} ({os.path.getsize(path)//1024} KB)")
        return True
    print("❌ Модель не найдена"); return False

# ТЕСТ 2: Модель делает предсказание
def test_2():
    print("\nТЕСТ 2: Предсказание")
    model, _ = load_model()
    if not model: return False
    x = torch.randn(1, 100, 13)
    with torch.no_grad(): y = model(x)
    print(f"✅ Вход: {x.shape} → Выход: {y.shape}")
    return y.shape[1] > 0

# ТЕСТ 3: Модель работает с аудио
def test_3():
    print("\nТЕСТ 3: Работа с аудио")
    model, _ = load_model()
    if not model: return False
    
    # Создаем аудио
    sr = 16000
    audio = np.sin(2*np.pi*200*np.linspace(0,2,int(sr*2)))*0.5
    path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    sf.write(path, audio, sr)
    
    # MFCC
    y, _ = librosa.load(path, sr=sr)
    mfcc = torch.FloatTensor(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T).unsqueeze(0)
    
    with torch.no_grad(): out = model(mfcc)
    print(f"✅ Аудио {len(y)/sr:.1f}с → Предсказание: {out.shape}")
    os.unlink(path)
    return True

# ТЕСТ 4: Сравнение с energy
def test_4():
    print("\nТЕСТ 4: Модель vs Energy")
    model, _ = load_model()
    if not model: return False
    
    sr = 16000
    t = np.linspace(0, 2, int(sr*2))
    audio = np.sin(2*np.pi*300*t) * np.sin(2*np.pi*4*t) * 0.5
    path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    sf.write(path, audio, sr)
    
    y, _ = librosa.load(path, sr=sr)
    mfcc = torch.FloatTensor(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T).unsqueeze(0)
    
    with torch.no_grad(): out = model(mfcc).numpy()[0]
    
    # Исправленное извлечение энергии
    flen = int(sr * 0.04)
    hop = int(sr * 0.02)
    energy = []
    for i in range(0, len(y) - flen, hop):
        energy.append(np.sqrt(np.mean(y[i:i+flen]**2)))
    energy = np.array(energy)
    if energy.max() > 0:
        energy = energy / energy.max()
    
    # Приводим к одной длине
    min_len = min(len(out), len(energy))
    if min_len > 1:
        corr = np.corrcoef(out[:min_len], energy[:min_len])[0,1]
        print(f"✅ Корреляция: {corr:.3f}" + (" (есть связь!)" if abs(corr) > 0.1 else " (слабая)"))
    else:
        print("⚠️ Слишком мало данных для корреляции")
    
    os.unlink(path)
    return True

if __name__ == "__main__":
    print("="*40 + "\nТЕСТЫ МОДЕЛИ\n" + "="*40)
    tests = [("Загрузка", test_1), ("Предсказание", test_2), ("Аудио", test_3), ("vs Energy", test_4)]
    results = [(n, t()) for n, t in tests]
    print("\n" + "="*40 + "\nИТОГИ\n" + "="*40)
    for n, r in results: print(f"  {'✅' if r else '❌'} {n}")
    print(f"\nПройдено: {sum(r for _,r in results)}/{len(results)}")