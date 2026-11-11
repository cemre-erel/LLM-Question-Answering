import chromadb
import ollama

from pathlib import Path
import fitz


import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_PATH = Path(r"C:\Users\Cemre\OneDrive\Masaüstü\LLMQuestionAnswering\data\Cemre_Erel_KenarTespiti.pdf")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


doc = fitz.open(PDF_PATH)
pages = []

for page_number, page in enumerate(doc, start=1):
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(image, lang="tur+eng")

    if text.strip():
        pages.append((page_number, text))

all_chunks = []

for page_number, text in pages:
    for chunk_number, chunk in enumerate(split_text(text), start=1):
        all_chunks.append({
            "page": page_number,
            "chunk": chunk_number,
            "text": chunk,
        })

print(f"PDF sayfası: {len(doc)}")
print(f"Metin bulunan sayfa: {len(pages)}")
print(f"Oluşturulan parça: {len(all_chunks)}")

for item in all_chunks[:3]:
    print(f"\n--- Sayfa {item['page']}, parça {item['chunk']} ---")
    print(item["text"])


client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="pdf_qa_ocr")

for item in all_chunks:
    embedding = ollama.embeddings(
        model="bge-m3",
        prompt=item["text"]
    )["embedding"]

    doc_id = f"sayfa-{item['page']}-parca-{item['chunk']}"

    collection.upsert(
        ids=[doc_id],
        documents=[item["text"]],
        embeddings=[embedding],
        metadatas=[{"page": item["page"], "chunk": item["chunk"]}]
    )

print(f"\nChroma'da kayıtlı parça: {collection.count()}")

# Deneme sorusu
questions = [
    "Kenar algılama nedir?",
    "Kenarlar görüntüde hangi sınırları tanımlar?",
    "Kenar algılama bölütleme ve nesne tanımaya nasıl yardımcı olur?",
    "Kenarlar doğru belirlenirse hangi temel özellikler ölçülebilir?",
    "Bu çalışmada kenar algılama teknikleri nasıl karşılaştırılmıştır?",
]

for question in questions:
    query_embedding = ollama.embeddings(
        model="bge-m3",
        prompt=question
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"[Sayfa {metadata['page']}]\n{document}"
        )

    context = "\n\n".join(context_parts)

    response = ollama.chat(
        model="gemma2:9b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Soruyu yalnızca verilen bağlama dayanarak yanıtla. "
                    "Bağlamda cevap yoksa bunu açıkça söyle. "
                    "Yanıtın sonunda kullandığın sayfa numaralarını belirt."
                ),
            },
            {
                "role": "user",
                "content": f"Bağlam:\n{context}\n\nSoru: {question}",
            },
        ],
        options={"temperature": 0},
    )

    print(f"\n\n{'=' * 60}")
    print(f"SORU: {question}")
    print("YANIT:")
    print(response["message"]["content"])
    print("GETİRİLEN SAYFALAR:")
    print(sorted({metadata["page"] for metadata in metadatas}))

query_embedding = ollama.embeddings(
    model="bge-m3",
    prompt=question
)["embedding"]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

context_parts = []
for document, metadata in zip(documents, metadatas):
    context_parts.append(
        f"[Sayfa {metadata['page']}]\n{document}"
    )

context = "\n\n".join(context_parts)

response = ollama.chat(
    model="gemma2:9b",
    messages=[
        {
            "role": "system",
            "content": (
                "Soruyu yalnızca verilen bağlama dayanarak yanıtla. "
                "Bağlamda cevap yoksa bunu açıkça söyle. "
                "Yanıtın sonunda kullandığın sayfa numaralarını belirt."
            ),
        },
        {
            "role": "user",
            "content": f"Bağlam:\n{context}\n\nSoru: {question}",
        },
    ],
    options={"temperature": 0},
)

print("\n--- LLM yanıtı ---")
print(response["message"]["content"])

print(f"\nSoru: {question}")
for i, document in enumerate(results["documents"][0], start=1):
    metadata = results["metadatas"][0][i - 1]
    print(f"\n--- Sonuç {i} | Sayfa {metadata['page']} ---")
    print(document)