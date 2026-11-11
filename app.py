
import streamlit as st
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import io
from openpyxl.styles import Alignment, Font
from rag_utils import (
    MAX_PDF_SIZE_MB,
    extract_pdf_chunks,
    index_pdf_chunks,
    retrieve_context,
)



st.set_page_config(
    page_title="Soru-Cevap",
    page_icon="🤖"
)

st.title("Soru-Cevap Uygulaması")

prompts = {
    "Normal": "Soruyu doğrudan ve anlaşılır şekilde cevapla.",
    "Detaylı": "Soruyu adım adım açıkla ve örnekler ver.",
    "Basit": "Soruyu yeni öğrenen birine basit bir dille açıkla."
}




def to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sonuçlar")
        ws = writer.sheets["Sonuçlar"]

        ws.column_dimensions["A"].width = 35
        ws.column_dimensions["B"].width = 12
        ws.column_dimensions["C"].width = 90

        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        for cell in ws[1]:
            cell.font = Font(bold=True)

    return buffer.getvalue()

BASE_RULE = (
    "You are a helpful assistant. "
    "Always reply in the same language as the user's latest message "
    "(Turkish question -> Turkish answer, English question -> English answer). "
    "Never use Chinese or any language the user did not write in. "
    "If you are not sure about a fact, say you are not sure instead of making it up."
)

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = Path(__file__).parent / "qwen3_1_7b_lora_adapter"


@st.cache_resource(show_spinner="Düşünüyor...")
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
    ).to("cuda")

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return tokenizer, model


def generate_answer(messages):
    tokenizer, model = load_local_model()

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ask_llm(question, prompt):
    messages = [
        {"role": "system", "content": f"{BASE_RULE} {prompt}"},
        {"role": "user", "content": question},
    ]
    return generate_answer(messages)


def ask_llm_stream(history, prompt):
    tokenizer, model = load_local_model()

    messages = [
        {"role": "system", "content": f"{BASE_RULE} {prompt}"}
    ]

    for message in history:
        if message.get("content") and message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"],
            })

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
        daemon=True,
    )
    thread.start()

    for text_chunk in streamer:
        yield text_chunk

    thread.join()

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "next_id" not in st.session_state:
    st.session_state.next_id = max(st.session_state.chats.keys(), default=0) + 1

if "current" not in st.session_state:
    st.session_state.current = None


def new_chat():
    cid = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.chats[cid] = {"title": "Yeni sohbet", "messages": []}
    st.session_state.current = cid


if st.session_state.current not in st.session_state.chats:
    if st.session_state.chats:
        st.session_state.current = list(st.session_state.chats)[-1]
    else:
        new_chat()


st.markdown("""
<style>
[class*="st-key-del_"] { opacity: 0; transition: opacity .15s; }
[class*="st-key-row_"]:hover [class*="st-key-del_"] { opacity: 1; }
[class*="st-key-del_"] button,
[class*="st-key-del_"] button * {
    color: #e5655b;
}
[class*="st-key-del_"] button {
    border: none;
    background: transparent;
}
[class*="st-key-del_"] button:hover {
    background: rgba(229, 101, 91, 0.12);
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:

    if st.button("➕ Yeni sohbet", use_container_width=True):
        new_chat()
        st.rerun()

    st.subheader("Sohbetler")
    for cid in reversed(list(st.session_state.chats)):
        item = st.session_state.chats[cid]
        with st.container(key=f"row_{cid}"):
            c_open, c_del = st.columns([5, 1])
            with c_open:
                if st.button(
                    item["title"],
                    key=f"open_{cid}",
                    use_container_width=True,
                    type="primary" if cid == st.session_state.current else "secondary"
                ):
                    st.session_state.current = cid
                    st.rerun()
            with c_del:
                if st.button("", key=f"del_{cid}", icon=":material/delete:"):
                    del st.session_state.chats[cid]
                    st.rerun()

tab_chat, tab_test = st.tabs(["💬 Sohbet", "🧪 Prompt Karşılaştırma Testi"])

with tab_chat:
    chat = st.session_state.chats[st.session_state.current]

    chat_area = st.container()

    col_prompt, col_input = st.columns([1, 4])
    with col_prompt:
        prompt_type = st.selectbox(
            "Prompt türü",
            list(prompts.keys()),
            label_visibility="collapsed"
        )
    with col_input:
        submission = st.chat_input(
         "Sorunu yaz...",
         accept_file=True,
         file_type=["pdf"],
         max_upload_size=MAX_PDF_SIZE_MB,
         key="chat_input_with_pdf",
        )

    question = submission.text if submission else None
    uploaded_files = submission.files if submission else []

    with chat_area:
        for msg_index, msg in enumerate(chat["messages"]):
            with st.chat_message(msg["role"]):
                if msg.get("content"):
                    st.write(msg["content"])

                for file_index, attachment in enumerate(msg.get("files", [])):
                    st.download_button(
                        label=f"📄 {attachment['name']}",
                        data=attachment["data"],
                        file_name=attachment["name"],
                        mime="application/pdf",
                        key=f"pdf-{st.session_state.current}-{msg_index}-{file_index}",
                    )

        if question or uploaded_files:
            try:
                attachments = []

                if uploaded_files:
                    uploaded_pdf = uploaded_files[0]
                    pdf_bytes = uploaded_pdf.getvalue()

                    with st.spinner("PDF okunuyor ve indeksleniyor..."):
                        chunks = extract_pdf_chunks(pdf_bytes)

                        if not chunks:
                            raise ValueError("PDF'den metin çıkarılamadı.")

                        collection_name, chunk_count = index_pdf_chunks(
                            pdf_bytes,
                            chunks,
                        )

                    st.session_state.rag_collection_name = collection_name
                    st.session_state.rag_source_name = uploaded_pdf.name

                    attachments.append({
                        "name": uploaded_pdf.name,
                        "data": pdf_bytes,
                    })

                first_message = not chat["messages"]
                user_content = question or ""

                chat["messages"].append({
                    "role": "user",
                    "content": user_content,
                    "files": attachments,
                })

                if first_message:
                    chat["title"] = (
                        question[:25]
                        if question
                        else attachments[0]["name"][:25]
                    )

                with st.chat_message("user"):
                    if question:
                        st.write(question)

                    for file_index, attachment in enumerate(attachments):
                        st.download_button(
                            label=f"📄 {attachment['name']}",
                            data=attachment["data"],
                            file_name=attachment["name"],
                            mime="application/pdf",
                            key=f"new-pdf-{st.session_state.current}-{file_index}",
                        )

                if question:
                    context = ""
                    source_pages = []
                    collection_name = st.session_state.get("rag_collection_name")

                    if collection_name:
                        context, source_pages = retrieve_context(
                            question,
                            collection_name,
                        )

                    answer_prompt = prompts[prompt_type]

                    if collection_name:
                        answer_prompt += (
                            "\n\nYanıtı yalnızca aşağıdaki PDF bağlamına göre ver. "
                            "Bağlamda yanıt yoksa bunu açıkça söyle.\n\n"
                            f"PDF bağlamı:\n{context}"
                        )

                    with st.chat_message("assistant"):
                        answer = st.write_stream(
                            ask_llm_stream(
                                chat["messages"],
                                answer_prompt,
                            )
                        )

                        if source_pages:
                            source_text = (
                                "Kaynak sayfa(lar): "
                                + ", ".join(map(str, source_pages))
                            )
                            st.caption(source_text)
                            answer = f"{answer}\n\n{source_text}"

                    chat["messages"].append({
                        "role": "assistant",
                        "content": answer,
                    })

                elif attachments:
                    confirmation = (
                        f"{attachments[0]['name']} işlendi "
                        f"({chunk_count} parça). Şimdi PDF hakkında soru sorabilirsin."
                    )

                    with st.chat_message("assistant"):
                        st.write(confirmation)

                    chat["messages"].append({
                        "role": "assistant",
                        "content": confirmation,
                    })

                if first_message:
                    st.rerun()

            except Exception as error:
                st.error(f"Hata: {error}")
with tab_test:
    questions = [
        "Türkiye'nin başkenti neresidir?",
        "Balıklar nerede yaşar?",
        "Buz eriyince ne olur?",
        "Haftada kaç gün vardır?",
        "Cumhuriyet bayramı ne zamandır?"
    ]

    if st.button("5 Soruyu Test Et"):
        results = []

        progress = st.progress(0)
        total = len(questions) * len(prompts)
        completed = 0

        for q in questions:
            for prompt_name, prompt_text in prompts.items():
                try:
                    answer = ask_llm(q, prompt_text)
                except Exception as error:
                    answer = f"Hata: {error}"

                results.append({
                    "Soru": q,
                    "Prompt": prompt_name,
                    "Cevap": answer
                })

                completed += 1
                progress.progress(completed / total)

        st.session_state.test_df = pd.DataFrame(results)
        st.success("15 test tamamlandı.")

    if "test_df" in st.session_state:
        df = st.session_state.test_df

        _, col_btn = st.columns([4, 1])
        with col_btn:
            st.download_button(
                "Excel indir",
                data=to_excel(df),
                file_name="prompt_test_sonuclari.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
                use_container_width=True
            )

        st.dataframe(df, use_container_width=True)