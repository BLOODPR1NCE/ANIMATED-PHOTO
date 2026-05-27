# dataset_generator_audio_text.py
import json, random, os, re
from pathlib import Path
from gtts import gTTS

TEMPLATES = {
    "tech_support": [
        "I can't connect to the Wi-Fi network at {location}. What should I do?",
        "My laptop keeps showing a blue screen with error code {error_code}. Please help.",
        "How do I reset my password for the {service} account?",
    ],
    "education": [
        "Can you explain the theory of {theory} in simple terms?",
        "What are the key differences between {concept_a} and {concept_b}?",
        "I need a summary of chapter {chapter} of the {subject} textbook.",
    ],
    "daily_info": [
        "What's the weather forecast for {city} tomorrow?",
        "Set a reminder for my meeting at {time} on {date}.",
        "Find me the best recipe for {dish}.",
    ]
}

FILLERS = {
    "location": ["home", "the office", "the cafe"], "error_code": ["0x0000007B", "0x800F0922"],
    "service": ["email", "corporate portal", "Jira"], "software": ["Python", "Photoshop", "Docker"],
    "browser": ["Chrome", "Firefox", "Edge"], "app": ["Instagram", "TikTok", "YouTube"],
    "theory": ["relativity", "evolution", "quantum mechanics"],
    "concept_a": ["machine learning", "deep learning"], "concept_b": ["AI", "neural networks"],
    "chapter": ["1", "2", "3"], "subject": ["history", "biology", "physics"],
    "city": ["Moscow", "New York", "Tokyo"], "time": ["10:00 AM", "2:30 PM"],
    "date": ["Friday", "March 15th"], "dish": ["pasta", "sushi", "borscht"],
    "business": ["bank", "gym", "pharmacy"], "genre": ["jazz", "rock", "pop"],
    "topic": ["space", "ancient Rome", "coffee"], "food": ["pizza", "sushi", "ice cream"],
    "language": ["French", "German", "Russian"], "amount": ["$45.50", "$120.00"]
}

def fill(template):
    for p in re.findall(r'\{(.*?)\}', template):
        if p in FILLERS: template = template.replace("{"+p+"}", random.choice(FILLERS[p]))
    return template

def generate_texts(n=200):
    data = []
    cats = list(TEMPLATES.keys())
    for cat in cats:
        for i in range(n//len(cats)):
            data.append({"id": f"{cat}_{i:04d}", "text": fill(random.choice(TEMPLATES[cat])), "category": cat, "language": "en"})
    return data

def synthesize(data, audio_dir):
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Синтез речи для {len(data)} текстов...")
    for i, item in enumerate(data):
        path = str(audio_dir / f"{item['id']}.mp3")
        if not os.path.exists(path):
            gTTS(text=item['text'], lang='en', slow=False).save(path)
        if (i+1) % 25 == 0: print(f"  ✓ {i+1}/{len(data)}")

def analyze(data, out_dir):
    counts = {}
    words = [w for item in data for w in item['text'].split()]
    for item in data: counts[item['category']] = counts.get(item['category'], 0) + 1
    
    meta = {
        "dataset_name": "Synthetic Speech Queries",
        "total": len(data), "categories": counts,
        "total_words": len(words), "unique_words": len(set(words)),
        "tts": "gTTS", "format": "MP3"
    }
    
    print(f"\n📊 АНАЛИЗ: {meta['total']} образцов, {meta['total_words']} слов, {meta['unique_words']} уникальных")
    for cat, cnt in counts.items(): print(f"  • {cat}: {cnt} ({cnt/len(data)*100:.0f}%)")
    
    with open(out_dir / "metadata.json", 'w') as f: json.dump(meta, f, indent=2)
    with open(out_dir / "transcripts" / "all.json", 'w') as f: json.dump(data, f, indent=2)
    print(f"✅ Сохранено в: {out_dir}")

if __name__ == "__main__":
    print("🎙️ ГЕНЕРАТОР ДАТАСЕТА")
    out = Path("./dataset_audio_text")
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    
    data = generate_texts(200)
    synthesize(data, out / "audio")
    analyze(data, out)