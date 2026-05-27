# api.py
import os, tempfile, subprocess
import numpy as np, cv2, librosa, dlib, imageio
import torch, torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

class LipSyncModel(nn.Module):
    def __init__(self, in_dim=13, hid=256, out_dim=136):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(), nn.Dropout(.2), nn.Linear(hid,hid), nn.ReLU())
        self.lstm = nn.LSTM(hid, hid, 2, batch_first=True, dropout=.2, bidirectional=True)
        self.dec = nn.Sequential(nn.Linear(hid*2,512), nn.ReLU(), nn.Dropout(.3), nn.Linear(512,256), nn.ReLU(), nn.Linear(256,out_dim), nn.Tanh())
    def forward(self, x):
        x = self.enc(x); x, _ = self.lstm(x)
        return self.dec(x.mean(1))

class AvatarService:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.model = None
        for mp in ['final_lipsync_model.pth', 'best_lipsync_model.pth']:
            if os.path.exists(mp):
                try:
                    ck = torch.load(mp, map_location='cpu')['model_state_dict']
                    out_dim = ck['dec.5.weight'].shape[0]
                    hid = ck['enc.0.weight'].shape[0]
                    self.model = LipSyncModel(13, hid, out_dim)
                    self.model.load_state_dict(ck)
                    self.model.eval()
                    break
                except: pass
    
    def _landmarks(self, img):
        faces = self.detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if not faces: return None
        l = self.predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), faces[0])
        return np.array([[l.part(i).x, l.part(i).y] for i in range(68)])
    
    def _energy(self, path, fps=30):
        y, sr = librosa.load(path, sr=16000)
        n = max(30, int(len(y)/sr * fps))
        e = [np.sqrt(np.mean(y[i:i+int(sr*.04)]**2)) for i in range(0, len(y)-int(sr*.04), int(sr*.02))]
        if e:
            e = np.clip(np.array(e)/max(e)*2, 0.1, 1.0)
            from scipy.ndimage import gaussian_filter1d as gf
            e = gf(e, sigma=2)
        idx = np.linspace(0, len(e)-1, n).astype(int) if len(e)>0 else range(n)
        return [float(e[i]) if i<len(e) else .5 for i in idx]
    
    def animate(self, img_path, aud_path, out_path):
        img = cv2.imread(img_path)
        src = self._landmarks(img)
        if src is None: return None
        
        energy = self._energy(aud_path)
        frames = []
        for e in energy:
            dst = src.copy()
            mo = min(.8, .1+e*1.2)
            for i in range(48,60): dst[i,1] = src[i,1] + (15 if i<54 else -15)*mo
            for i in range(60,68): dst[i,1] = src[i,1] + (12 if i<64 else -12)*mo
            
            h, w = img.shape[:2]
            gx, gy = np.meshgrid(np.arange(w,dtype=np.float32), np.arange(h,dtype=np.float32))
            mx, my = gx.copy(), gy.copy()
            for s, d in zip(src, dst):
                disp = d-s
                wgt = np.clip(np.exp(-((gx-s[0])**2+(gy-s[1])**2)/(2*50**2)), 0, 1)
                mx += disp[0]*wgt; my += disp[1]*wgt
            frames.append(cv2.cvtColor(cv2.remap(img.copy(), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), cv2.COLOR_BGR2RGB))
        
        try:
            from moviepy.editor import ImageSequenceClip, AudioFileClip
            v = ImageSequenceClip(frames, fps=25); a = AudioFileClip(aud_path)
            v.set_audio(a).write_videofile(out_path, codec='libx264', audio_codec='aac', fps=25, verbose=False, logger=None)
        except:
            t = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            imageio.mimsave(t, frames, fps=25, codec='libx264')
            subprocess.run(['ffmpeg','-y','-i',t,'-i',aud_path,'-c:v','copy','-c:a','aac','-shortest',out_path], capture_output=True)
            os.unlink(t)
        return out_path

async def tts(text, voice="male"):
    p = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
    try:
        import edge_tts
        await edge_tts.Communicate(text=text, voice={"male":"ru-RU-DmitryNeural","female":"ru-RU-SvetlanaNeural"}[voice]).save(p)
        if os.path.getsize(p)>500: return p
    except: pass
    return None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
svc = AvatarService()

@app.post("/generate")
async def gen(photo: UploadFile = File(...), text: str = Form("Привет!"), voice: str = Form("male")):
    t = []
    try:
        img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name; t.append(img)
        with open(img,'wb') as f: f.write(await photo.read())
        aud = await tts(text, voice)
        if not aud: return JSONResponse({"error":"TTS"}, 500)
        t.append(aud)
        out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        r = svc.animate(img, aud, out)
        return FileResponse(r, media_type="video/mp4") if r else JSONResponse({"error":"Failed"}, 500)
    finally:
        for f in t:
            try: os.unlink(f)
            except: pass

if __name__ == "__main__":
    print("🚀 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)