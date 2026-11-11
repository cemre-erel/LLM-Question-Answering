# Büyük Dil Modellerinde Fine-Tuning ve RAG

Bu proje, büyük dil modellerinde RAG ve fine-tuning yaklaşımlarını uygulamalı olarak incelemek amacıyla geliştirilmiştir.

## Proje bölümleri

### RAG tabanlı PDF soru-cevap sistemi
- PDF dosyalarından metin çıkarma ve metni parçalara ayırma
- `bge-m3` ile metin parçalarını vektörleştirme
- Vektörleri ChromaDB’de saklama ve soruya uygun parçaları getirme
- Ollama üzerinden yanıt üretme
- Streamlit arayüzünde PDF yükleme ve soru-cevap

### LoRA fine-tuning denemesi
- Temel model: `Qwen/Qwen3-1.7B`
- Yöntem: LoRA
- Veri kümesi: `merve/turkish_instructions`
- Eğitim için kullanılan örnek: 898
- Eğitim öncesi ve sonrası çıktılar beş soru üzerinden karşılaştırılmıştır.

## Dosyalar

- `app.py`: Streamlit uygulaması
- `rag_utils.py`: RAG ve belge işleme yardımcıları
- `finetune_qwen3.py`: LoRA eğitim kodu
- `compare_qwen3.py`: Eğitim öncesi ve sonrası çıktıları karşılaştırma
- `baseline_instruct.py`: Temel model çıktıları için deneme kodu
- `pdf_test.py`: PDF metin çıkarma denemesi

## Kurulum

pip install streamlit transformers peft datasets accelerate chromadb ollama pymupdf pytesseract pillow pandas openpyxl

### Kullanılan modeller

- **Qwen/Qwen3-1.7B:** LoRA fine-tuning ve sohbet uygulamasındaki yanıt üretimi için temel model.
- **bge-m3:** PDF parçalarını ve kullanıcı sorularını vektörleştirmek için Ollama üzerinden kullanılır.
- **gemma2:9b:** Önceki RAG denemelerinde yanıt modeli olarak kullanılmıştır. Güncel `app.py` içindeki yanıt modeli Qwen3-1.7B ve LoRA adaptörüdür.

Fine-tuning sırasında Hugging Face, Qwen3-1.7B modelini ve `merve/turkish_instructions` veri kümesini indirir. RAG özellikleri için Ollama’da `bge-m3` ve `gemma2:9b` modellerinin kurulu olması gerekir.
