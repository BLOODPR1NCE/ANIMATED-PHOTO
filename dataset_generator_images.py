# dataset_generator_image.py
import os, json, time, random, threading
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_face(gender):
    try:
        url = f"https://randomuser.me/api/portraits/{'men' if gender=='male' else 'women'}/{random.randint(0,99)}.jpg"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 1000:
            img = Image.open(BytesIO(r.content)).resize((512,512), Image.Resampling.LANCZOS)
            if img.mode != 'RGB': img = img.convert('RGB')
            return img, gender
    except: pass
    return None, None

def generate_dataset(target_gb=0.5, out_dir="./gender_faces"):
    out = Path(out_dir)
    for f in ["male", "female"]: (out / "images" / f).mkdir(parents=True, exist_ok=True)
    
    target = target_gb * 1024**3
    total_size, total_photos = 0, 0
    counts = {"male": 0, "female": 0}
    metadata = []
    lock = threading.Lock()
    
    print(f"Генерация датасета: {target_gb} ГБ")
    pbar = tqdm(total=target, unit='B', unit_scale=True, desc="Загрузка")
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        while total_size < target:
            futures = [ex.submit(download_face, random.choice(["male","female"])) for _ in range(8)]
            for f in as_completed(futures):
                img, gender = f.result()
                if img:
                    path = out / "images" / gender / f"{gender}_{int(time.time()*1e6)}.jpg"
                    img.save(path, 'JPEG', quality=92)
                    size = os.path.getsize(path)
                    
                    with lock:
                        total_photos += 1; total_size += size
                        counts[gender] += 1
                        metadata.append({"file": str(path.relative_to(out)), "folder": gender})
                        pbar.update(size)
                        pbar.set_postfix({'Фото': total_photos, 'M': counts['male'], 'F': counts['female']})
    
    pbar.close()
    
    analysis = {
        "total": total_photos, "size_gb": round(total_size/1024**3, 2),
        "male": counts['male'], "female": counts['female'],
        "source": "randomuser.me", "resolution": "512x512"
    }
    with open(out / "metadata.json", 'w') as f: json.dump(analysis, f, indent=2)
    with open(out / "labels.json", 'w') as f: json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Готово! {total_photos:,} фото, {total_size/1024**3:.2f} ГБ")
    print(f"👨 {counts['male']:,} мужчин, 👩 {counts['female']:,} женщин")

if __name__ == "__main__":
    generate_dataset(target_gb=0.5)