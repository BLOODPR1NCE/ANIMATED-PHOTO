# api.py - ИСПРАВЛЕННЫЕ ПУТИ К МОДЕЛИ
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import librosa
import dlib
import subprocess
import imageio
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class LipSyncModel(torch.nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256, output_dim=136):
        super().__init__()
        self.audio_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim), torch.nn.Tanh()
        )
        
    def forward(self, x):
        x = self.audio_encoder(x)
        x, _ = self.lstm(x)
        return self.decoder(x.mean(dim=1))

class FaceWarper:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = self._get_predictor_path()
        self.predictor = dlib.shape_predictor(self.predictor_path) if self.predictor_path else None
    
    @staticmethod
    def _get_predictor_path() -> Optional[str]:
        possible_paths = [
            "./data/shape_predictor_68_face_landmarks.dat",
            "shape_predictor_68_face_landmarks.dat",
            os.path.expanduser("~/shape_predictor_68_face_landmarks.dat")
        ]
        return next((p for p in possible_paths if os.path.exists(p)), None)
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.predictor: return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces: return None
        landmarks = self.predictor(gray, faces[0])
        return np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
    
    def warp_face(self, image, src_landmarks, dst_landmarks):
        if src_landmarks is None or dst_landmarks is None: return image
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x, map_y = grid_x.astype(np.float32), grid_y.astype(np.float32)
        
        for src, dst in zip(src_landmarks, dst_landmarks):
            displacement = dst - src
            dist = np.sqrt((grid_x - src[0])**2 + (grid_y - src[1])**2)
            weight = np.clip(np.exp(-dist**2 / (2 * 50**2)), 0, 1)
            map_x += displacement[0] * weight
            map_y += displacement[1] * weight
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def create_lip_animation(self, image, audio_energy):
        src_landmarks = self.detect_landmarks(image)
        if src_landmarks is None: return [image] * len(audio_energy)
        
        frames = []
        for energy in audio_energy:
            frame = image.copy()
            current = src_landmarks.copy()
            mouth_open = min(0.8, 0.1 + energy * 1.2)
            
            for i in range(48, 60):
                offset = 15 if i < 54 else -15
                current[i][1] = src_landmarks[i][1] + offset * mouth_open
            for i in range(60, 68):
                offset = 12 if i < 64 else -12
                current[i][1] = src_landmarks[i][1] + offset * mouth_open
            if np.random.random() < 0.05:
                for i in range(36, 48):
                    current[i][1] = src_landmarks[i][1] + 8
            
            frames.append(self.warp_face(frame, src_landmarks, current))
        return frames


class DigitalAvatarService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_warper = FaceWarper()
        self.model = None
        
        model_paths = [
            'best_lipsync_model.pth',           # train_lipsync_model.py сохраняет сюда
            'final_lipsync_model.pth',
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    
                    if 'decoder.5.weight' in state_dict:
                        out_dim = state_dict['decoder.5.weight'].shape[0]
                        hid_dim = state_dict['audio_encoder.0.weight'].shape[0]
                        self.model = LipSyncModel(13, hid_dim, out_dim).to(self.device)
                        self.model.load_state_dict(state_dict)
                    
                    if self.model:
                        self.model.eval()
                        print(f"✅ Модель загружена из {model_path}")
                        break
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки {model_path}: {e}")
        
        if not self.model:
            print("⚠️ Модель не загружена, использую energy-based анимацию")
    
    def extract_audio_energy(self, audio_path: str) -> List[float]:
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            frame_length, hop_length = int(sr * 0.04), int(sr * 0.02)
            energy = [np.sqrt(np.mean(y[i:i+frame_length]**2)) for i in range(0, len(y) - frame_length, hop_length)]
            
            if energy:
                energy = np.array(energy)
                if energy.max() > 0: energy = energy / energy.max()
                energy = np.clip(energy, 0.1, 1.0)
                from scipy.ndimage import gaussian_filter1d
                energy = gaussian_filter1d(energy, sigma=2)
            
            num_frames = 30
            indices = np.linspace(0, len(energy) - 1, num_frames) if len(energy) > 0 else np.zeros(num_frames)
            return [float(energy[int(idx)]) if 0 <= int(idx) < len(energy) else 0.5 for idx in indices]
        except:
            return [0.5] * 30
    
    def generate_animation(self, image_path: str, audio_path: str, output_path: str) -> Optional[str]:
        image = cv2.imread(image_path)
        audio_energy = self.extract_audio_energy(audio_path)
        frames = self.face_warper.create_lip_animation(image, audio_energy) or [image] * len(audio_energy)
        
        try:
            resized_frames = []
            for frame in frames:
                if frame.shape[1] > 800:
                    scale = 800 / frame.shape[1]
                    frame = cv2.resize(frame, (800, int(frame.shape[0] * scale)))
                resized_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            try:
                from moviepy.editor import ImageSequenceClip, AudioFileClip
                video = ImageSequenceClip(resized_frames, fps=25)
                audio = AudioFileClip(audio_path)
                final = video.set_audio(audio)
                final.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=25, verbose=False, logger=None)
                video.close(); audio.close()
            except:
                temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                imageio.mimsave(temp, resized_frames, fps=25, codec='libx264')
                subprocess.run(['ffmpeg','-y','-i',temp,'-i',audio_path,'-c:v','copy','-c:a','aac','-shortest',output_path], capture_output=True)
                os.unlink(temp)
            
            return output_path
        except Exception as e:
            print(f"Ошибка сохранения: {e}")
            return None


async def text_to_speech(text: str, voice: str = "male") -> Optional[str]:
    path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
    try:
        import edge_tts
        voices = {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural", "male_low": "ru-RU-DmitryNeural"}
        v = voices.get(voice, "ru-RU-DmitryNeural")
        rate = "-30%" if "low" in voice else "+0%"
        await edge_tts.Communicate(text=text, voice=v, rate=rate).save(path)
        if os.path.exists(path) and os.path.getsize(path) > 500: return path
    except: pass
    try:
        from gtts import gTTS
        gTTS(text=text, lang='ru').save(path)
        if os.path.getsize(path) > 500: return path
    except: pass
    return None


app = FastAPI(title="DigitalAvatar API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
service = DigitalAvatarService()

@app.post("/generate")
async def generate_animation(photo: UploadFile = File(...), text: str = Form("Привет!"), voice: str = Form("male")):
    temp_files = []
    try:
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        temp_files.append(img_path)
        with open(img_path, 'wb') as f: f.write(await photo.read())
        
        audio_path = await text_to_speech(text, voice)
        if not audio_path: return JSONResponse({"error": "Не удалось создать речь"}, 500)
        temp_files.append(audio_path)
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        temp_files.append(output_path)
        
        result = service.generate_animation(img_path, audio_path, output_path)
        if result and os.path.exists(result):
            return FileResponse(result, media_type="video/mp4", filename="avatar.mp4")
        return JSONResponse({"error": "Ошибка создания анимации"}, 500)
    finally:
        for f in temp_files[:2]:
            try:
                if os.path.exists(f): os.unlink(f)
            except: pass

if __name__ == "__main__":
    print("🚀 DigitalAvatar API - http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)