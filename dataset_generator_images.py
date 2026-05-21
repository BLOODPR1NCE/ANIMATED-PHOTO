# dataset_generator_gender.py
import os
import json
import time
from pathlib import Path
from typing import Dict, List
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

class GenderDatasetGenerator:
    """Генератор датасета реальных фото с распределением по ПОЛУ."""
    
    def __init__(self, output_dir: str = "./gender_face_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.folders = {
            "male": "Мужчины",
            "female": "Женщины"
        }
        
        for folder in self.folders:
            (self.images_dir / folder).mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()
        self.total_photos = 0
        self.total_size = 0
        self.metadata = []
        self.folder_counts = {"male": 0, "female": 0}
        
    def download_face(self, gender: str) -> Dict:
        """Скачивает фото конкретного пола."""
        try:
            img_id = random.randint(0, 99)
            
            if gender == "male":
                url = f"https://randomuser.me/api/portraits/men/{img_id}.jpg"
            else:
                url = f"https://randomuser.me/api/portraits/women/{img_id}.jpg"
            
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            
            if response.status_code == 200 and len(response.content) > 1000:
                img = Image.open(BytesIO(response.content))
                
                # Увеличиваем
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Сохраняем
                timestamp = int(time.time() * 1000000)
                filename = f"{gender}_{timestamp}.jpg"
                filepath = self.images_dir / gender / filename
                
                img.save(filepath, 'JPEG', quality=92)
                
                file_size = os.path.getsize(filepath)
                
                return {
                    "file": str(filepath.relative_to(self.output_dir)),
                    "folder": gender,
                    "gender": gender,
                    "size_bytes": file_size
                }
        except:
            pass
        
        return None
    
    def generate_dataset(self, target_gb: float = 0.5 ): #сколько ГБ надо
        """Генерирует датасет."""
        target_bytes = target_gb * 1024 * 1024 * 1024
        
        print("="*60)
        print("ГЕНЕРАТОР ДАТАСЕТА ПО ПОЛУ")
        print("="*60)
        print(f"Источник: randomuser.me (реальные фото)")
        print(f"Цель: {target_gb} ГБ")
        print(f"Категории: Мужчины / Женщины")
        print(f"Разрешение: 512x512 JPEG")
        print("="*60)
        print()
        
        time.sleep(1)
        
        start_time = time.time()
        workers = 8
        
        pbar = tqdm(
            total=target_bytes,
            desc="Загрузка фото",
            unit='B',
            unit_scale=True,
            colour='green'
        )
        
        last_print = 0
        
        while self.total_size < target_bytes:
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                
                for _ in range(workers):
                    gender = random.choice(["male", "female"])
                    future = executor.submit(self.download_face, gender)
                    futures.append(future)
                
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result:
                        with self.lock:
                            self.metadata.append(result)
                            self.total_photos += 1
                            self.total_size += result['size_bytes']
                            self.folder_counts[result['folder']] += 1
                            pbar.update(result['size_bytes'])
                        
                        current_gb = self.total_size / (1024**3)
                        elapsed = time.time() - start_time
                        
                        pbar.set_postfix({
                            'Фото': f"{self.total_photos:,}",
                            'ГБ': f"{current_gb:.2f}",
                            'M': self.folder_counts['male'],
                            'F': self.folder_counts['female']
                        })
            
            if self.total_photos - last_print >= 1000:
                last_print = self.total_photos
                current_gb = self.total_size / (1024**3)
                elapsed = time.time() - start_time
                
                print(f"\n{'='*40}")
                print(f"📸 {self.total_photos:,} фото | 💾 {current_gb:.2f} ГБ | ⏱️ {elapsed/60:.1f} мин")
                print(f"👨 Мужчины: {self.folder_counts['male']:,}")
                print(f"👩 Женщины: {self.folder_counts['female']:,}")
            
            if self.total_photos % 5000 == 0:
                self.save_checkpoint()
        
        pbar.close()
        self.save_results(start_time)
        
        return self.metadata
    
    def save_checkpoint(self):
        """Сохранение прогресса."""
        try:
            checkpoint = {
                "total": self.total_photos,
                "size_gb": round(self.total_size / (1024**3), 2),
                "male": self.folder_counts['male'],
                "female": self.folder_counts['female']
            }
            
            with open(self.output_dir / "checkpoint.json", 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except:
            pass
    
    def save_results(self, start_time: float):
        """Сохранение результатов."""
        
        elapsed = time.time() - start_time
        
        analysis = {
            "dataset_name": "Gender Classification Face Dataset",
            "description": "Датасет реальных фото людей для классификации пола.",
            "version": "1.0",
            "total_photos": self.total_photos,
            "total_size_gb": round(self.total_size / (1024**3), 2),
            "total_size_mb": round(self.total_size / (1024**2), 1),
            "avg_photo_kb": round(self.total_size / self.total_photos / 1024, 1) if self.total_photos else 0,
            "resolution": "512x512",
            "format": "JPEG (quality 92)",
            "source": "randomuser.me (real photos)",
            "categories": {
                "male": "Мужчины",
                "female": "Женщины"
            },
            "distribution": {
                "male": self.folder_counts['male'],
                "female": self.folder_counts['female'],
                "male_percent": round(self.folder_counts['male'] / self.total_photos * 100, 1),
                "female_percent": round(self.folder_counts['female'] / self.total_photos * 100, 1)
            },
            "generation_time_minutes": round(elapsed / 60, 1),
            "photos_per_minute": round(self.total_photos / (elapsed / 60), 1) if elapsed > 0 else 0,
            "applications": [
                "Классификация пола (Gender Classification)",
                "Распознавание лиц (Face Recognition)",
                "Детекция лиц (Face Detection)",
                "Обучение нейросетей",
                "Аугментация данных"
            ]
        }
        
        # Сохраняем
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Разметка (только файлы)
        labels = [
            {"file": m["file"], "folder": m["folder"]}
            for m in self.metadata
        ]
        
        with open(self.output_dir / "labels.json", 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2)
        
        # Вывод
        print("\n" + "="*60)
        print("✅ ДАТАСЕТ ГОТОВ!")
        print("="*60)
        print(f"📸 Фото: {self.total_photos:,}")
        print(f"💾 Размер: {self.total_size / (1024**3):.2f} ГБ")
        print(f"⏱️  Время: {elapsed/60:.1f} мин")
        print(f"\n📁 Распределение:")
        print(f"   👨 Мужчины: {self.folder_counts['male']:,} ({analysis['distribution']['male_percent']}%)")
        print(f"   👩 Женщины: {self.folder_counts['female']:,} ({analysis['distribution']['female_percent']}%)")
        print(f"\n📍 {self.output_dir.absolute()}")


if __name__ == "__main__":
    import random
    
    print("="*60)
    print("ГЕНЕРАТОР ДАТАСЕТА ПО ПОЛУ")
    print("="*60)
    print()
    
    gen = GenderDatasetGenerator()
    gen.generate_dataset(target_gb=0.5)