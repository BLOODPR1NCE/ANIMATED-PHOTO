# gui.py - ВАШ ИСХОДНЫЙ GUI + ВЫБОР ГОЛОСА
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import os
import threading
import shutil
from PIL import Image, ImageTk

class DigitalAvatarGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DigitalAvatar - Анимация лица")
        self.root.geometry("800x650")
        self.root.configure(bg='#f0f0f0')
        self.photo_path: str = None
        self.api_url = "http://localhost:8000"
        self.setup_ui()
    
    def setup_ui(self):
        tk.Label(self.root, text="🎭 DigitalAvatar", font=("Arial", 28, "bold"), bg='#f0f0f0', fg='#333').pack(pady=20)
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=10, padx=20)
        
        btn_frame = tk.Frame(self.root, bg='#f0f0f0')
        btn_frame.pack(pady=20)
        self.photo_btn = tk.Button(btn_frame, text="📸 Загрузить фото", command=self.load_photo, bg='#4CAF50', fg='white', font=("Arial", 11, "bold"), padx=20, pady=10, cursor='hand2')
        self.photo_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="📷 С камеры", command=self.use_camera, bg='#2196F3', fg='white', font=("Arial", 11, "bold"), padx=20, pady=10, cursor='hand2').pack(side=tk.LEFT, padx=10)
        
        info_frame = tk.Frame(self.root, bg='#f0f0f0', relief=tk.GROOVE, bd=1)
        info_frame.pack(pady=10, padx=20, fill='x')
        self.photo_label = tk.Label(info_frame, text="📷 Фото: не загружено", font=("Arial", 10), bg='#f0f0f0', fg='gray', pady=5)
        self.photo_label.pack(anchor='w', padx=10)
        
        preview_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=2)
        preview_frame.pack(pady=10, padx=20, fill='both', expand=True)
        tk.Label(preview_frame, text="Предпросмотр", font=("Arial", 12, "bold"), bg='white').pack(pady=5)
        self.image_label = tk.Label(preview_frame, bg='#e0e0e0', text="Загрузите фото", font=("Arial", 10), width=50, height=15, relief=tk.SUNKEN)
        self.image_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Текст
        text_frame = tk.Frame(self.root, bg='#f0f0f0')
        text_frame.pack(pady=5, padx=20, fill='x')
        tk.Label(text_frame, text="Текст для озвучки:", font=("Arial", 10), bg='#f0f0f0').pack(anchor='w')
        self.text_input = tk.Text(text_frame, height=3, font=("Arial", 11), wrap='word')
        self.text_input.pack(fill='x', pady=5)
        self.text_input.insert('1.0', 'Привет! Это мой цифровой аватар!')
        
        # Голос
        voice_frame = tk.Frame(self.root, bg='#f0f0f0')
        voice_frame.pack(pady=5, padx=20, fill='x')
        tk.Label(voice_frame, text="Голос:", font=("Arial", 10), bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        self.voice_var = tk.StringVar(value="Мужской")
        ttk.Combobox(voice_frame, textvariable=self.voice_var, values=["Мужской", "Женский", "Низкий мужской"], width=20).pack(side=tk.LEFT, padx=5)
        
        gen_frame = tk.Frame(self.root, bg='#f0f0f0')
        gen_frame.pack(pady=20)
        self.generate_btn = tk.Button(gen_frame, text="🎬 СОЗДАТЬ АНИМАЦИЮ 🎬", command=self.generate_animation, bg='#FF9800', fg='white', font=("Arial", 14, "bold"), padx=30, pady=15, cursor='hand2', state=tk.DISABLED)
        self.generate_btn.pack()
    
    def load_photo(self):
        path = filedialog.askopenfilename(title="Выберите фото", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.photo_path = path
            self.photo_label.config(text=f"📷 Фото: {os.path.basename(path)}", fg="green")
            self.generate_btn.config(state=tk.NORMAL)
            self._preview_image(path)
    
    def use_camera(self):
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.photo_path = "camera_photo.jpg"
                cv2.imwrite(self.photo_path, frame)
                self.photo_label.config(text="📷 Фото: с камеры", fg="green")
                self.generate_btn.config(state=tk.NORMAL)
                self._preview_image(self.photo_path)
        except:
            messagebox.showerror("Ошибка", "Установите opencv-python")
    
    def _preview_image(self, file_path):
        try:
            img = Image.open(file_path)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except:
            pass
    
    def generate_animation(self):
        text = self.text_input.get('1.0', 'end-1c').strip()
        if not text:
            messagebox.showwarning("Внимание", "Введите текст!")
            return
        
        voice_map = {"Мужской": "male", "Женский": "female", "Низкий мужской": "male_low"}
        voice = voice_map.get(self.voice_var.get(), "male")
        
        self.generate_btn.config(state=tk.DISABLED, text="⏳ Создание...")
        threading.Thread(target=self._generate_thread, args=(text, voice), daemon=True).start()
    
    def _generate_thread(self, text, voice):
        try:
            with open(self.photo_path, 'rb') as f:
                response = requests.post(f'{self.api_url}/generate', files={'photo': f}, data={'text': text, 'voice': voice}, timeout=120)
            
            if response.status_code == 200:
                temp_path = "temp_avatar.mp4"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                self.root.after(0, self._save_dialog, temp_path)
            else:
                error = response.json().get('error', 'Ошибка сервера')
                self.root.after(0, lambda: messagebox.showerror("Ошибка", error))
        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", "Сервер не запущен!\nЗапустите: python api.py"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL, text="🎬 СОЗДАТЬ АНИМАЦИЮ 🎬"))
    
    def _save_dialog(self, temp_path):
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 видео", "*.mp4")], initialfile="avatar.mp4")
        if save_path:
            shutil.copy(temp_path, save_path)
            messagebox.showinfo("Готово!", f"Видео сохранено:\n{save_path}")
        try:
            os.unlink(temp_path)
        except:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().theme_use('clam')
    app = DigitalAvatarGUI(root)
    root.mainloop()