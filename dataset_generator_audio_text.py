# dataset_generator_audio_text.py - ОЧИЩЕННАЯ ВЕРСИЯ
import json
import random
import os
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
from gtts import gTTS

print("✅ Используется gTTS для синтеза речи")


class AudioTextDatasetGenerator:
    """Генератор датасета аудио + текст"""
    
    def __init__(self, output_dir: str = "./generated_dataset_audio_text"):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.transcript_dir = self.output_dir / "transcripts"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        self.templates = {
            "tech_support": [
                "I can't connect to the Wi-Fi network at {location}. What should I do?",
                "My laptop keeps showing a blue screen with error code {error_code}. Please help.",
                "How do I reset my password for the {service} account?",
                "The printer at {location} is not responding. Can you troubleshoot?",
                "I need help installing {software} on my computer.",
                "My phone battery drains too fast. Any tips?",
                "How do I clear the cache in {browser}?",
                "The {app} on my phone keeps crashing. How to fix it?",
                "Can you help me set up two-factor authentication?",
                "I'm having issues with the VPN connection to the office network."
            ],
            "education": [
                "Can you explain the theory of {theory} in simple terms?",
                "What are the key differences between {concept_a} and {concept_b}?",
                "I need a summary of chapter {chapter} of the {subject} textbook.",
                "Help me solve this math problem: {math_problem}",
                "What is the history of the {historical_event}?",
                "Can you provide an example of {grammar_rule} in a sentence?",
                "What are the main themes in '{book_title}'?",
                "How do I write a proper essay on {essay_topic}?",
                "Explain the {scientific_process} process step by step.",
                "What's the formula for {formula_name}?"
            ],
            "daily_info": [
                "What's the weather forecast for {city} tomorrow?",
                "Set a reminder for my meeting at {time} on {date}.",
                "Find me the best recipe for {dish}.",
                "What are the opening hours of the {business} near me?",
                "Play some {genre} music from the {decade}s.",
                "Tell me a fun fact about {topic}.",
                "How many calories are in a portion of {food}?",
                "Translate '{phrase}' into {language}.",
                "What's the latest news on {news_topic}?",
                "Calculate the tip for a {amount} dollar bill."
            ]
        }
        
        self.fillers = {
            "location": ["home", "the office", "the cafe", "the library", "the airport"],
            "error_code": ["0x0000007B", "0x800F0922", "0x8024402F", "0x80070002"],
            "service": ["email", "corporate portal", "HR system", "Jira", "Slack"],
            "software": ["Python 3.10", "Adobe Photoshop", "Microsoft Teams", "Docker"],
            "browser": ["Chrome", "Firefox", "Safari", "Edge"],
            "app": ["Instagram", "TikTok", "YouTube", "Spotify", "Uber"],
            "theory": ["relativity", "evolution", "quantum mechanics", "string theory"],
            "concept_a": ["machine learning", "deep learning"],
            "concept_b": ["artificial intelligence", "neural networks"],
            "chapter": [str(i) for i in range(1, 11)],
            "subject": ["history", "biology", "physics", "literature", "economics"],
            "math_problem": ["2x + 5 = 15", "integral of x^2 dx", "derivative of sin(x)"],
            "historical_event": ["French Revolution", "Moon Landing", "Fall of Berlin Wall"],
            "grammar_rule": ["Past Perfect", "Subjunctive mood", "Conditional sentences"],
            "book_title": ["1984", "To Kill a Mockingbird", "The Great Gatsby"],
            "essay_topic": ["climate change", "social media impact", "renewable energy"],
            "scientific_process": ["photosynthesis", "digestion", "combustion", "DNA replication"],
            "formula_name": ["quadratic", "Pythagorean theorem", "E=mc^2", "speed"],
            "city": ["Moscow", "New York", "Tokyo", "London", "Berlin"],
            "time": ["10:00 AM", "2:30 PM", "6:45 PM"],
            "date": ["Friday", "March 15th", "the 1st of April"],
            "dish": ["pasta carbonara", "chicken tikka masala", "sushi", "borscht"],
            "business": ["bank", "gym", "pharmacy", "supermarket", "library"],
            "genre": ["jazz", "rock", "pop", "classical", "hip-hop"],
            "decade": ["80", "90", "2000", "2010"],
            "topic": ["space", "ancient Rome", "octopuses", "coffee"],
            "food": ["pizza", "sushi roll", "avocado toast", "ice cream"],
            "phrase": ["Good morning, how are you?", "Thank you very much", "Where is the train station?"],
            "language": ["French", "German", "Spanish", "Japanese", "Russian"],
            "news_topic": ["technology", "sports", "politics", "environment", "business"],
            "amount": ["$45.50", "$120.00", "$15.75", "$68.20"]
        }

    def _fill_template(self, template: str) -> str:
        """Заполняет шаблон случайными значениями."""
        filled_text = template
        placeholders = re.findall(r'\{(.*?)\}', template)
        for placeholder in placeholders:
            if placeholder in self.fillers:
                filled_text = filled_text.replace("{" + placeholder + "}", random.choice(self.fillers[placeholder]))
        return filled_text

    def generate_texts(self, num_samples: int = 200) -> List[Dict]:
        """Генерирует список текстов и метаданных"""
        generated_data = []
        print(f"Генерация {num_samples} текстовых запросов...")
        
        categories = list(self.templates.keys())
        samples_per_category = num_samples // len(categories)
        
        for category in categories:
            for i in range(samples_per_category):
                template = random.choice(self.templates[category])
                final_text = self._fill_template(template)
                generated_data.append({
                    "id": f"{category}_{i:04d}",
                    "text": final_text,
                    "category": category,
                    "language": "en"
                })
        
        return generated_data

    def synthesize_audio(self, text_data: List[Dict]) -> None:
        """Синтезирует аудио из текстов через gTTS."""
        print(f"Синтез речи для {len(text_data)} текстов...")
        success_count = 0
        
        for i, item in enumerate(text_data):
            try:
                output_path = str(self.audio_dir / f"{item['id']}.mp3")
                
                if os.path.exists(output_path):
                    success_count += 1
                    continue
                
                tts = gTTS(text=item['text'], lang='en', slow=False)
                tts.save(output_path)
                success_count += 1
                
                if (i + 1) % 25 == 0:
                    print(f"  ✓ Синтезировано {i + 1}/{len(text_data)}...")
                    
            except Exception as e:
                print(f"  ✗ Ошибка синтеза {item['id']}: {e}")
        
        print(f"✅ Успешно синтезировано: {success_count}/{len(text_data)}")

    def analyze_and_save(self, data: List[Dict]) -> None:
        """Анализирует датасет и сохраняет метаданные."""
        metadata = {
            "dataset_name": "Synthetic Speech Queries Dataset",
            "description": "Датасет синтезированной речи для обучения моделей распознавания речи и NLP.",
            "total_samples": len(data),
            "categories": {},
            "avg_text_length": 0.0,
            "avg_word_length": 0.0,
            "language": "en",
            "tts_engine": "gTTS",
            "audio_format": "MP3"
        }
        
        lengths = []
        word_lengths = []
        total_words = 0
        
        for item in data:
            category = item['category']
            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1
            
            words = item['text'].split()
            lengths.append(len(words))
            word_lengths.extend([len(w) for w in words])
            total_words += len(words)
        
        metadata['avg_text_length'] = round(np.mean(lengths), 2) if lengths else 0
        metadata['avg_word_length'] = round(np.mean(word_lengths), 2) if word_lengths else 0
        metadata['total_words'] = total_words
        metadata['unique_words'] = len(set(w for item in data for w in item['text'].lower().split()))
        
        # Баланс категорий
        category_counts = list(metadata['categories'].values())
        if len(category_counts) > 1:
            std_cat = np.std(category_counts)
            mean_cat = np.mean(category_counts)
            metadata['category_balance_score'] = round(1.0 - (std_cat / mean_cat) if mean_cat > 0 else 0, 3)
        
        # Вывод анализа
        print("\n" + "="*50)
        print("📊 АНАЛИЗ ДАТАСЕТА (Аудио + Текст)")
        print("="*50)
        print(f"Всего образцов: {metadata['total_samples']}")
        print(f"Всего слов: {total_words}")
        print(f"Уникальных слов: {metadata['unique_words']}")
        print(f"Средняя длина текста: {metadata['avg_text_length']} слов")
        print(f"Средняя длина слова: {metadata['avg_word_length']} символов")
        print(f"\nРаспределение по категориям:")
        for cat, count in metadata['categories'].items():
            print(f"  • {cat}: {count} ({count/len(data)*100:.1f}%)")
        print(f"\nСферы применения:")
        print("  • Распознавание речи (Speech-to-Text)")
        print("  • Классификация запросов техподдержки")
        print("  • Обучение голосовых ассистентов")
        print("  • Анализ тональности и интентов")
        print("  • Генерация ответов на вопросы")
        
        # Сохранение файлов
        transcripts_path = self.transcript_dir / "all_transcripts.json"
        with open(transcripts_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Датасет сохранен в: {self.output_dir}")
        print(f"   📁 {transcripts_path}")
        print(f"   📁 {metadata_path}")

if __name__ == "__main__":
    print("="*50)
    print("🎙️ ГЕНЕРАТОР ДАТАСЕТА: АУДИО + ТЕКСТ")
    print("="*50)
    
    generator = AudioTextDatasetGenerator()
    text_data = generator.generate_texts(200)
    generator.synthesize_audio(text_data)
    generator.analyze_and_save(text_data)