# gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests, os, threading, shutil
from PIL import Image, ImageTk

class AvatarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 AnimatedPhoto")
        self.root.geometry("600x550")
        self.root.configure(bg='#f0f0f0')
        self.photo_path = None
        self.api = "http://localhost:8000"
        
        # UI
        tk.Label(root, text="🎭 AnimatedPhoto", font=("Arial", 22, "bold"), bg='#f0f0f0').pack(pady=10)
        self.preview = tk.Label(root, text="Сделайте фото с камеры", bg='#e0e0e0', width=50, height=12)
        self.preview.pack(pady=10, padx=20)
        
        tk.Button(root, text="📷 С камеры", command=self.camera, bg='#2196F3', fg='white', font=("Arial", 11), padx=20, pady=8).pack(pady=5)
        
        tk.Label(root, text="Текст:", bg='#f0f0f0').pack(anchor='w', padx=20)
        self.text = tk.Text(root, height=3, font=("Arial", 11))
        self.text.pack(pady=5, padx=20, fill='x')
        self.text.insert('1.0', 'Привет! Это мой аватар!')
        
        # Голос
        f = tk.Frame(root, bg='#f0f0f0'); f.pack(pady=5)
        tk.Label(f, text="Голос:", bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.voice = tk.StringVar(value="Мужской")
        ttk.Combobox(f, textvariable=self.voice, values=["Мужской", "Женский"], width=15).pack(side=tk.LEFT)
        
        self.btn = tk.Button(root, text="🎬 Создать видео", command=self.generate, bg='#FF9800', fg='white', font=("Arial", 13, "bold"), padx=30, pady=10, state=tk.DISABLED)
        self.btn.pack(pady=15)
        self.status = tk.Label(root, text="Готов", bg='#f0f0f0', fg='gray'); self.status.pack()
    
    def camera(self):
        try:
            import cv2
            cap = cv2.VideoCapture(0);
            ret, frame = cap.read();
            cap.release()
            if ret:
                self.photo_path = "photo.jpg";
                cv2.imwrite(self.photo_path, frame)
                self.btn.config(state=tk.NORMAL)
                img = Image.open(self.photo_path);
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.preview.config(image=photo, text='');
                self.preview.image = photo
        except:
            messagebox.showerror("Ошибка", "pip install opencv-python")
    
    def generate(self):
        txt = self.text.get('1.0', 'end-1c').strip()
        if not txt: return messagebox.showwarning("Внимание", "Введите текст!")
        
        voice_map = {"Мужской": "male", "Женский": "female"}
        voice = voice_map.get(self.voice.get(), "male")
        
        self.btn.config(state=tk.DISABLED, text="⏳ Создание...")
        threading.Thread(target=self._gen, args=(txt, voice), daemon=True).start()
    
    def _gen(self, txt, voice):
        try:
            with open(self.photo_path, 'rb') as f:
                r = requests.post(f'{self.api}/generate', files={'photo': f}, data={'text': txt, 'voice': voice}, timeout=120)
            if r.status_code == 200:
                temp = "temp.mp4"
                with open(temp, 'wb') as f: f.write(r.content)
                self.root.after(0, self._save, temp)
            else:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", r.json().get('error','Ошибка')))
        finally:
            self.root.after(0, lambda: self.btn.config(state=tk.NORMAL, text="🎬 Создать видео"))
    
    def _save(self, temp):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")], initialfile="avatar.mp4")
        if path:
            shutil.copy(temp, path)
            messagebox.showinfo("Готово!", f"Сохранено:\n{path}")
        try: os.unlink(temp)
        except: pass

if __name__ == "__main__":
    AvatarGUI(tk.Tk())
    tk.mainloop()