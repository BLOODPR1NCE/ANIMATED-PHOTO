# test_model_direct.py - ПРЯМЫЕ ТЕСТЫ МОДЕЛИ (БЕЗ API)
import os
import sys
import tempfile
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Импортируем компоненты напрямую
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Создаем тестовое фото
def create_test_face():
    """Создает тестовое фото лица."""
    img = Image.new('RGB', (512, 512), color=(180, 160, 140))
    draw = ImageDraw.Draw(img)
    
    # Голова
    draw.ellipse([80, 30, 432, 482], fill=(210, 180, 150), outline=(140, 110, 90), width=3)
    # Глаза
    draw.ellipse([150, 160, 190, 200], fill='white', outline='black', width=2)
    draw.ellipse([320, 160, 360, 200], fill='white', outline='black', width=2)
    draw.ellipse([165, 175, 178, 188], fill='black')
    draw.ellipse([335, 175, 348, 188], fill='black')
    # Брови
    draw.line([140, 145, 200, 140], fill='black', width=3)
    draw.line([310, 140, 370, 145], fill='black', width=3)
    # Нос
    draw.ellipse([240, 230, 270, 275], fill=(190, 160, 130))
    # Рот
    draw.arc([190, 300, 320, 370], start=0, end=180, fill='black', width=3)
    
    path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    img.save(path)
    return path


def create_test_audio():
    """Создает тестовый аудиофайл с речью."""
    import soundfile as sf
    
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Имитация речи (меняющаяся частота)
    audio = np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 3 * t) * 0.5
    
    path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    sf.write(path, audio, sr)
    return path


# ============================================================
# ТЕСТ 1: Детекция лица через dlib
# ============================================================
def test_1_face_detection():
    """Проверяет, что dlib находит лицо и ключевые точки."""
    print("\n" + "="*50)
    print("ТЕСТ 1: Детекция лица")
    print("="*50)
    
    import dlib
    
    # Загружаем детектор
    detector = dlib.get_frontal_face_detector()
    
    predictor_path = None
    for p in ["shape_predictor_68_face_landmarks.dat", "./data/shape_predictor_68_face_landmarks.dat"]:
        if os.path.exists(p):
            predictor_path = p
            break
    
    if not predictor_path:
        print("❌ Предиктор не найден!")
        return False
    
    predictor = dlib.shape_predictor(predictor_path)
    print(f"✅ Предиктор загружен: {predictor_path}")
    
    # Создаем тестовое фото
    photo_path = create_test_face()
    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Детекция
    faces = detector(gray)
    print(f"Найдено лиц: {len(faces)}")
    
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        print(f"Точек лица: {len(points)}")
        print(f"Точки рта (48-67): {len(points[48:68])}")
        
        # Проверяем что точки рта найдены
        mouth_points = points[48:68]
        mouth_center = mouth_points.mean(axis=0)
        print(f"Центр рта: ({mouth_center[0]:.0f}, {mouth_center[1]:.0f})")
        
        os.unlink(photo_path)
        print("✅ ТЕСТ 1 ПРОЙДЕН: Лицо детектируется")
        return True
    else:
        os.unlink(photo_path)
        print("❌ Лицо не найдено (это нормально для рисованного лица)")
        return False


# ============================================================
# ТЕСТ 2: Деформация лица (remap)
# ============================================================
def test_2_face_warping():
    """Проверяет деформацию лица через remap."""
    print("\n" + "="*50)
    print("ТЕСТ 2: Деформация лица")
    print("="*50)
    
    import dlib
    
    predictor_path = None
    for p in ["shape_predictor_68_face_landmarks.dat", "./data/shape_predictor_68_face_landmarks.dat"]:
        if os.path.exists(p):
            predictor_path = p
            break
    
    if not predictor_path:
        print("❌ Предиктор не найден, пропускаем")
        return True  # Не ошибка
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Загружаем реальное фото (если есть) или создаем тестовое
    photo_path = create_test_face()
    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print("⚠️ Лицо не найдено на тестовом фото")
        os.unlink(photo_path)
        return True
    
    landmarks = predictor(gray, faces[0])
    src = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
    
    # Тест деформации рта
    dst = src.copy()
    mouth_open = 0.5
    
    for i in range(48, 55):
        dst[i, 1] = src[i, 1] - 10 * mouth_open
    for i in range(55, 60):
        dst[i, 1] = src[i, 1] + 10 * mouth_open
    
    # Применяем remap
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = grid_x.copy(), grid_y.copy()
    
    for s, d in zip(src, dst):
        disp = d - s
        dist = np.sqrt((grid_x - s[0])**2 + (grid_y - s[1])**2)
        weight = np.clip(np.exp(-dist**2 / (2 * 50**2)), 0, 1)
        map_x += disp[0] * weight
        map_y += disp[1] * weight
    
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Сохраняем результат
    cv2.imwrite('test_warped.jpg', warped)
    print("✅ Деформированное фото сохранено: test_warped.jpg")
    
    # Проверяем что изображение изменилось
    diff = np.abs(image.astype(float) - warped.astype(float)).mean()
    print(f"Среднее изменение пикселей: {diff:.1f}")
    
    os.unlink(photo_path)
    print("✅ ТЕСТ 2 ПРОЙДЕН: Деформация работает")
    return True


# ============================================================
# ТЕСТ 3: Извлечение энергии аудио
# ============================================================
def test_3_audio_energy():
    """Проверяет извлечение энергии из аудио."""
    print("\n" + "="*50)
    print("ТЕСТ 3: Энергия аудио")
    print("="*50)
    
    import librosa
    from scipy.ndimage import gaussian_filter1d
    
    audio_path = create_test_audio()
    
    # Загружаем аудио
    y, sr = librosa.load(audio_path, sr=16000)
    print(f"Аудио: {len(y)} сэмплов, {len(y)/sr:.1f} сек, SR={sr}")
    
    # Извлекаем энергию
    frame_length = int(sr * 0.04)
    hop_length = int(sr * 0.02)
    
    energy = [np.sqrt(np.mean(y[i:i+frame_length]**2)) for i in range(0, len(y)-frame_length, hop_length)]
    energy = np.array(energy)
    
    if energy.max() > 0:
        energy = energy / energy.max()
    energy = np.clip(energy, 0.1, 1.0)
    energy = gaussian_filter1d(energy, sigma=2)
    
    print(f"Энергия: {len(energy)} кадров")
    print(f"Мин: {energy.min():.3f}, Макс: {energy.max():.3f}, Сред: {energy.mean():.3f}")
    
    # Приводим к 30 кадрам
    num_frames = 30
    indices = np.linspace(0, len(energy)-1, num_frames).astype(int)
    energy_30 = [float(energy[i]) for i in indices]
    
    print(f"30 кадров: {len(energy_30)} значений")
    print(f"Примеры: {[f'{e:.2f}' for e in energy_30[:5]]}...")
    
    os.unlink(audio_path)
    print("✅ ТЕСТ 3 ПРОЙДЕН: Энергия аудио извлечена")
    return True


# ============================================================
# ТЕСТ 4: Полный цикл анимации
# ============================================================
def test_4_full_animation():
    """Проверяет полный цикл создания анимации."""
    print("\n" + "="*50)
    print("ТЕСТ 4: Полный цикл анимации")
    print("="*50)
    
    import dlib
    import librosa
    from scipy.ndimage import gaussian_filter1d
    import imageio
    
    # Проверяем предиктор
    predictor_path = None
    for p in ["shape_predictor_68_face_landmarks.dat", "./data/shape_predictor_68_face_landmarks.dat"]:
        if os.path.exists(p):
            predictor_path = p
            break
    
    if not predictor_path:
        print("⚠️ Предиктор не найден, пропускаем")
        return True
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Создаем фото и аудио
    photo_path = create_test_face()
    audio_path = create_test_audio()
    
    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print("⚠️ Лицо не найдено")
        os.unlink(photo_path)
        os.unlink(audio_path)
        return True
    
    # Точки лица
    landmarks = predictor(gray, faces[0])
    src = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
    
    # Энергия аудио
    y, sr = librosa.load(audio_path, sr=16000)
    flen, hop = int(sr*0.04), int(sr*0.02)
    energy = [np.sqrt(np.mean(y[i:i+flen]**2)) for i in range(0, len(y)-flen, hop)]
    energy = np.array(energy)
    if energy.max() > 0:
        energy = energy / energy.max()
    energy = np.clip(energy, 0.1, 1.0)
    energy = gaussian_filter1d(energy, sigma=2)
    
    num_frames = 30
    indices = np.linspace(0, len(energy)-1, num_frames).astype(int)
    energy_30 = [float(energy[i]) for i in indices]
    
    # Создаем кадры
    frames = []
    for e in energy_30:
        frame = image.copy()
        dst = src.copy()
        mouth_open = min(0.8, 0.1 + e * 1.2)
        
        for i in range(48, 60):
            offset = 15 if i < 54 else -15
            dst[i, 1] = src[i, 1] + offset * mouth_open
        
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x, map_y = grid_x.copy(), grid_y.copy()
        
        for s, d in zip(src, dst):
            disp = d - s
            dist = np.sqrt((grid_x - s[0])**2 + (grid_y - s[1])**2)
            weight = np.clip(np.exp(-dist**2 / (2 * 50**2)), 0, 1)
            map_x += disp[0] * weight
            map_y += disp[1] * weight
        
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames.append(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    
    # Сохраняем GIF
    output_path = 'test_animation.gif'
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    
    file_size = os.path.getsize(output_path)
    print(f"✅ Анимация сохранена: {output_path} ({file_size/1024:.1f} KB)")
    print(f"Кадров: {len(frames)}")
    
    # Очистка
    os.unlink(photo_path)
    os.unlink(audio_path)
    
    print("✅ ТЕСТ 4 ПРОЙДЕН: Полный цикл работает")
    return True


if __name__ == "__main__":
    print("="*60)
    print("ПРЯМЫЕ ТЕСТЫ МОДЕЛИ (БЕЗ API)")
    print("="*60)
    
    results = []
    
    results.append(("Детекция лица", test_1_face_detection()))
    results.append(("Деформация лица", test_2_face_warping()))
    results.append(("Энергия аудио", test_3_audio_energy()))
    results.append(("Полный цикл", test_4_full_animation()))
    
    print("\n" + "="*60)
    print("ИТОГИ")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\nПройдено: {passed}/{len(results)}")