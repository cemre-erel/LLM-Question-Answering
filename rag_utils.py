import pymupdf
import hashlib
import chromadb
import ollama
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

MAX_PDF_SIZE_MB = 20
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "bge-m3"

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


def extract_pdf_chunks(pdf_bytes):
    max_size_bytes = MAX_PDF_SIZE_MB * 1024 * 1024

    if not pdf_bytes:
        raise ValueError("Yüklenen dosya boş.")

    if len(pdf_bytes) > max_size_bytes:
        raise ValueError(
            f"PDF en fazla {MAX_PDF_SIZE_MB} MB olabilir."
        )

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    all_chunks = []

    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            alpha=False,
        )
        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )
        text = pytesseract.image_to_string(image, lang="tur+eng")

        for chunk_number, chunk in enumerate(split_text(text), start=1):
            all_chunks.append({
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk,
            })

    doc.close()
    return all_chunks

def index_pdf_chunks(pdf_bytes, chunks):
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    collection_name = f"pdf_{file_hash}"

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    for item in chunks:
        embedding = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=item["text"],
        )["embedding"]

        chunk_id = f"page-{item['page']}-chunk-{item['chunk']}"

        collection.upsert(
            ids=[chunk_id],
            documents=[item["text"]],
            embeddings=[embedding],
            metadatas=[{
                "page": item["page"],
                "chunk": item["chunk"],
            }],
        )

    return collection_name, collection.count()


def retrieve_context(question, collection_name, n_results=3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=collection_name)

    count = collection.count()
    if count == 0:
        return "", []

    query_embedding = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question,
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    pages = set()

    for document, metadata in zip(documents, metadatas):
        context_parts.append(f"[Sayfa {metadata['page']}]\n{document}")
        pages.add(metadata["page"])

    context = "\n\n".join(context_parts)
    return context, sorted(pages)