"""LangChain + Ollama ile araç kullanan basit AI Agent."""
import argparse
import os
import re
import time

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError

import tools as T

SYSTEM_PROMPT = (
    "Sen yardımcı bir asistansın. Kullanıcı hangi dilde yazdıysa o dilde cevap ver. "
    "Elindeki araçları yalnızca gerektiğinde kullan: hesaplama için hesapla, "
    "ansiklopedik bilgi için wikipedia_ara, uzun bir metni özetlemek için metin_ozetle, "
    "yüklenen PDF ile ilgili sorular için pdf_ara. Selamlaşma ve basit sohbette araç kullanma. "
    "Bir istek hem bir kaynaktan bilgi bulmayı hem de bulunan sayı üzerinde işlem yapmayı "
    "gerektiriyorsa önce ilgili bilgi aracını, ardından hesapla aracını çağır; aritmetik işlemi "
    "kendin zihinden yapma. Kullanıcı açıkça hesaplama istediğinde hesapla aracını kullan. "
    "Araç sonucunda olmayan bilgiyi uydurma. Kullanıcı özetlenecek metni vermediyse metni iste."
)


def build_agent(model="qwen3:4b", pdf=None, temperature=0, pdf_collection=None):
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
    if pdf:
        if not T.RAG_OK:
            raise RuntimeError("rag_utils.py bulunamadı; agent dosyalarını RAG klasörüne koyun.")
        pdf_collection, _ = T.index_pdf(pdf)
    tool_list = T.bind_tools(llm, pdf_collection=pdf_collection)
    return create_agent(llm, tools=tool_list, system_prompt=SYSTEM_PROMPT)


def _clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def run(agent, question, max_steps=8, history=None):
    """Soruyu çalıştırır; cevap, kullanılan araçlar, adım izi ve süreyi döndürür."""
    t0 = time.time()
    messages = []
    for item in history or []:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max_steps * 2 + 1},
        )
        msgs = result["messages"]
        answer = _clean(msgs[-1].content)
    except GraphRecursionError:
        return {"answer": "(adım sınırı aşıldı)", "tools": [], "trace": [],
                "llm_calls": max_steps, "seconds": round(time.time() - t0, 1)}

    used, trace = [], []
    for m in msgs:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                used.append(c["name"])
                trace.append(f"ÇAĞRI  {c['name']}({c['args']})")
        elif m.type == "tool":
            trace.append(f"SONUÇ  {str(m.content)[:200]}")
    return {
        "answer": answer, "tools": used, "trace": trace,
        "llm_calls": sum(1 for m in msgs if m.type == "ai"),
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--pdf", default=None, help="Araç olarak kullanılacak PDF yolu")
    args = ap.parse_args()

    agent = build_agent(args.model, args.pdf)
    print(f"Agent hazır ({args.model}). Çıkmak için 'q'.")
    while True:
        q = input("\nSoru: ").strip()
        if q.lower() in {"q", "quit", "çık"}:
            break
        r = run(agent, q)
        for line in r["trace"]:
            print("  ", line)
        print(f"\nCevap: {r['answer']}\n[{r['llm_calls']} model çağrısı, {r['seconds']} sn]")
