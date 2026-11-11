import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-1.7B"

questions = [
    "Bir iş görüşmesini yeni bitirdim. İşverene pozisyona olan ilgimi belirten kısa ve profesyonel bir mesaj yaz.",
    "Şehirlerde ağaç dikmenin üç faydasını maddeler halinde açıkla.",
    "Şu cümleyi edilgen yapıdan etken yapıya çevir: Yeni proje müdür tarafından tamamlandı.",
    "Kitap okumanın öğrencilere sağlayabileceği üç katkıyı kısaca yaz.",
    "Bir ekip arkadaşından toplantıyı başka bir güne almasını nazikçe isteyen kısa bir e-posta yaz.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to("cuda")
model.eval()

for i, question in enumerate(questions, start=1):
    messages = [{"role": "user", "content": question}]

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
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print(f"\n--- Soru {i} ---")
    print(question)
    print("Yanıt:", answer)