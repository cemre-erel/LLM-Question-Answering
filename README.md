# LLM Soru-Cevap ve AI Agent

Bu proje iki kullanım sunar:

- Streamlit arayüzü: yerel Qwen3-1.7B LoRA modelini kullanır; sohbet, PDF ile RAG ve prompt karşılaştırması sağlar.
- LangChain agent: Ollama üzerinden araç seçer; hesaplama, Wikipedia araması, özetleme ve isteğe bağlı PDF araması yapar.

## Kurulum

Python 3.10 veya üzeri ve Ollama gerekir. Proje klasöründe:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Ollama'yı çalıştırıp modelleri indirin:

~~~powershell
ollama pull qwen3:4b
ollama pull bge-m3
~~~

bge-m3, PDF RAG gömmeleri için kullanılır. Taranmış PDF'lerde Tesseract OCR ve Türkçe/İngilizce dil verileri gereklidir. Tesseract standart konumda değilse PowerShell'de TESSERACT_CMD ortam değişkenini tesseract.exe dosyanızın yoluna ayarlayın. Metin katmanı bulunan PDF'lerde OCR gerekmez.

## Çalıştırma

Streamlit arayüzü (üst bölümde “Yanıt motoru” seçicisinden Yerel LoRA veya AI Agent seç):

~~~powershell
streamlit run app.py
~~~

Araç kullanan agent:

~~~powershell
python agent.py --model qwen3:4b
~~~

Agent'a PDF aracı eklemek için:

~~~powershell
python agent.py --model qwen3:4b --pdf data/Cemre_Erel_KenarTespiti.pdf
~~~

Agent testleri:

~~~powershell
python run_tests.py --models qwen3:4b
python run_tests.py --models qwen3:4b --pdf data/Cemre_Erel_KenarTespiti.pdf --pdf-soru "PDF'in konusu nedir?"
~~~

Test çıktısı varsayılan olarak results.csv dosyasına yazılır. Önceki sonuçları korumak için --out yeni_sonuclar.csv kullanın.

## Notlar

- Yerel agent modeli ve bge-m3 Ollama servisine, Streamlit arayüzündeki LoRA modeli Hugging Face model dosyalarına ihtiyaç duyar.
- PDF indeksleri proje klasöründeki chroma_db altında saklanır.
- CPU'da arayüz çalışabilir ancak yanıtlar GPU'ya göre belirgin biçimde yavaş olabilir.
- Çok adımlı araç kullanımı modelin tool-calling başarısına bağlıdır; araç izi ve yanıtlar birlikte değerlendirilmelidir.
- Streamlit içindeki AI Agent modu Ollama'da qwen3:4b modelini kullanır. Farklı bir model için uygulamayı başlatmadan önce AGENT_MODEL ortam değişkenini model etiketiyle ayarlayın.
