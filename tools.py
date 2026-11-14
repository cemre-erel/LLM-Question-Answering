"""Agent araçları. rag_utils.py ile aynı klasöre koyun (PDF aracı için)."""
import ast
import hashlib
import operator
import re

import requests
from langchain_core.tools import tool

_state = {"llm": None, "pdf_collection": None}

try:
    from rag_utils import (
        CHROMA_PATH, extract_pdf_chunks, index_pdf_chunks, retrieve_context,
    )
    RAG_OK = True
except Exception:  # rag_utils veya bağımlılıkları yoksa PDF aracı kapalı kalır
    RAG_OK = False


def set_llm(llm):
    _state["llm"] = llm


def index_pdf(path):
    """PDF'i indeksler. Aynı PDF daha önce indekslendiyse OCR'ı atlar."""
    with open(path, "rb") as f:
        data = f.read()
    name = f"pdf_{hashlib.sha256(data).hexdigest()[:16]}"

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        if client.get_collection(name).count() > 0:
            _state["pdf_collection"] = name
            return name, "önbellekten"
    except Exception:
        pass

    chunks = extract_pdf_chunks(data)
    if not chunks:
        raise ValueError(
            "PDF'den metin çıkarılamadı. Dosyanın okunabilir olduğundan ve "
            "taranmış PDF ise Tesseract OCR'ın kurulu olduğundan emin olun."
        )
    name, count = index_pdf_chunks(data, chunks)
    _state["pdf_collection"] = name
    return name, count


# ---------- hesapla ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Üs çok büyük")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Desteklenmeyen ifade")


@tool
def hesapla(ifade: str) -> str:
    """Matematiksel bir ifadeyi hesaplar. Örnek: '15 * 24 + 7' veya '0.15 * 85000000'.
    Yalnızca sayılar ve + - * / ** % operatörleri desteklenir. Yüzde için ondalık kullan (%15 = 0.15)."""
    try:
        temiz = ifade.replace("^", "**").replace("×", "*").replace("÷", "/")
        tree = ast.parse(temiz.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("İfade çok karmaşık")
        sonuc = _eval(tree.body)
        if isinstance(sonuc, (int, float)) and abs(sonuc) > 1e100:
            raise ValueError("Sonuç izin verilen sınırı aşıyor")
        return str(sonuc)
    except ZeroDivisionError:
        return "Hata: sıfıra bölme yapılamaz."
    except Exception as e:
        return f"Hata: ifade hesaplanamadı ({e})."


# ---------- wikipedia ----------
_HEADERS = {"User-Agent": "ai-agent-odev/1.0 (egitim amacli)"}


def _wiki(sorgu, dil):
    r = requests.get(
        f"https://{dil}.wikipedia.org/w/api.php",
        params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": sorgu, "gsrlimit": 1, "prop": "extracts",
            "exintro": 1, "explaintext": 1, "exchars": 1200,
        },
        headers=_HEADERS, timeout=10,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        return None
    p = next(iter(pages.values()))
    return f"{p.get('title')}: {p.get('extract', '').strip()}"


@tool
def wikipedia_ara(sorgu: str) -> str:
    """Vikipedi'de bir konuyu arar ve giriş özetini döndürür. Kişi, yer, kavram gibi
    ansiklopedik bilgiler için kullan. Sorgu kısa olmalı (örn. 'Ankara', 'Alan Turing')."""
    try:
        sonuc = _wiki(sorgu, "tr") or _wiki(sorgu, "en")
        return sonuc or "Vikipedi'de sonuç bulunamadı."
    except Exception as e:
        return f"Hata: Vikipedi'ye ulaşılamadı ({e})."


# ---------- özetleme ----------
@tool
def metin_ozetle(metin: str) -> str:
    """Kullanıcının verdiği uzun bir metni 2-3 cümleyle özetler. 'metin' alanına özetlenecek
    metnin tamamını yaz."""
    llm = _state["llm"]
    if llm is None:
        return "Hata: model tanımlı değil."
    istek = (
        "Aşağıdaki metni Türkçe olarak en fazla 3 cümleyle özetle. "
        "Yalnızca metinde olanları kullan.\n\n" + metin
    )
    cevap = llm.invoke(istek).content
    return re.sub(r"<think>.*?</think>", "", cevap, flags=re.S).strip()


# ---------- pdf ----------
@tool
def pdf_ara(soru: str) -> str:
    """Kullanıcının yüklediği PDF içinde arama yapar ve soruyla ilgili bölümleri
    sayfa numaralarıyla döndürür. PDF ile ilgili sorularda kullan."""
    if not RAG_OK:
        return "Hata: PDF araması kullanılamıyor."
    if not _state["pdf_collection"]:
        return "Hata: henüz PDF yüklenmedi."
    baglam, sayfalar = retrieve_context(soru, _state["pdf_collection"])
    if not baglam:
        return "PDF içinde ilgili bölüm bulunamadı."
    return f"{baglam}\n\n(Kaynak sayfalar: {', '.join(map(str, sayfalar))})"


def bind_tools(llm, pdf_collection=None):
    """Create tools bound to this agent's model and PDF, avoiding shared session state."""
    @tool("metin_ozetle")
    def summarize_text(metin: str) -> str:
        """Kullanıcının verdiği metni Türkçe olarak en fazla 3 cümleyle özetler."""
        request = (
            "Aşağıdaki metni Türkçe olarak en fazla 3 cümleyle özetle. "
            "Yalnızca metinde olanları kullan.\n\n" + metin
        )
        answer = llm.invoke(request).content
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()

    bound = [hesapla, wikipedia_ara, summarize_text]
    if pdf_collection:
        if not RAG_OK:
            raise RuntimeError("PDF araması için rag_utils.py ve bağımlılıkları gerekli.")

        @tool("pdf_ara")
        def search_pdf(soru: str) -> str:
            """Kullanıcının yüklediği PDF içinde arama yapar, ilgili bölümleri ve sayfaları döndürür."""
            context, pages = retrieve_context(soru, pdf_collection)
            if not context:
                return "PDF içinde ilgili bölüm bulunamadı."
            return f"{context}\n\n(Kaynak sayfalar: {', '.join(map(str, pages))})"

        bound.append(search_pdf)
    return bound
