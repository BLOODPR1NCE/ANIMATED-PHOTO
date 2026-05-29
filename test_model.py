# test_model.py
import os, sys, tempfile, numpy as np, torch, librosa, soundfile as sf
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_model():
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

# ТЕСТ 1: Файл модели существует
def test_1():
    for p in ['final_lipsync_model.pth', 'best_lipsync_model.pth']:
        if os.path.exists(p):
            size_kb = os.path.getsize(p) // 1024
            return True, f"Файл: {p} ({size_kb} KB)"
    return False, "Модель не найдена"

# ТЕСТ 2: Модель загружается и выдает правильную размерность
def test_2():
    model, path = load_model()
    if not model: return False, "Не загрузилась"
    
    # Подаем случайные данные
    x = torch.randn(1, 100, 13)  # 1 батч, 100 кадров, 13 MFCC
    with torch.no_grad():
        y = model(x)
    
    # Проверяем что выход = 136 (68 точек × 2 координаты)
    ok = y.shape[1] == 136
    return ok, f"Вход: {x.shape} → Выход: {y.shape}"

# ТЕСТ 3: Модель работает с реальным аудио
def test_3():
    model, _ = load_model()
    if not model: return False, "Нет модели"
    
    # Создаем тестовый звук (2 секунды, 440 Гц)
    sr = 16000
    audio = np.sin(2*np.pi*440*np.linspace(0,2,int(sr*2)))*0.5
    path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    sf.write(path, audio, sr)
    
    # Извлекаем MFCC
    y, _ = librosa.load(path, sr=sr)
    mfcc = torch.FloatTensor(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        out = model(mfcc)
    
    os.unlink(path)
    return True, f"Аудио 2с → Предсказано {out.shape[1]} точек"

# ТЕСТ 4: Разные входы дают РАЗНЫЕ выходы
def test_4():
    model, _ = load_model()
    if not model: return False, "Нет модели"
    
    # Два РАЗНЫХ случайных входа
    x1 = torch.randn(1, 50, 13)
    x2 = torch.randn(1, 50, 13)  # другие числа
    
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    
    # Выходы должны отличаться
    diff = (y1 - y2).abs().mean().item()
    different = diff > 0.01
    
    return different, f"Разница выходов: {diff:.4f} {'✓' if different else '✗'}"

def save_xml(results):
    """Сохраняет результаты тестов в XML."""
    passed = sum(1 for _, (s, _) in results if s)
    total = len(results)
    
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="LipSync Tests" tests="{total}" passed="{passed}" time="{datetime.now().isoformat()}">
'''
    for i, (name, (success, detail)) in enumerate(results):
        xml += f'  <testcase name="Тест {i+1}: {name}" status="{"passed" if success else "failed"}">\n'
        xml += f'    <system-out>{detail}</system-out>\n'
        xml += '  </testcase>\n'
    xml += '</testsuite>'
    
    with open("test-results.xml", 'w', encoding='utf-8') as f:
        f.write(xml)
    print("✅ test-results.xml сохранен")

if __name__ == "__main__":
    print("="*45)
    print("ТЕСТЫ LipSync МОДЕЛИ")
    print("="*45)
    
    tests = [
        ("Файл модели", test_1),
        ("Размерность", test_2),
        ("Аудио", test_3),
        ("Разные входы", test_4),
    ]
    
    results = []
    for name, func in tests:
        ok, msg = func()
        results.append((name, (ok, msg)))
        print(f"  {'✅' if ok else '❌'} {name}: {msg}")
    
    passed = sum(1 for _, (ok, _) in results if ok)
    print(f"\nПройдено: {passed}/{len(results)}")
    
    save_xml(results)
    exit(0 if passed == 4 else 1)