import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "qwen3_1_7b_lora_adapter"
MAX_LENGTH = 256

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("merve/turkish_instructions", split="train")
dataset = dataset.shuffle(seed=42).select(range(1000))

dataset = dataset.filter(
    lambda x: (x["talimat"] or "").strip()
    != '"Her ne kadar..." ile başlayan bir cümle kurun.'
)
print("Uyumsuz örnek filtrelendikten sonraki sayı:", len(dataset))

def encode_example(example):
    instruction = (example["talimat"] or "").strip()
    extra_input = (example[" giriş"] or "").strip()
    answer = (example[" çıktı"] or "").strip()

    user_text = instruction
    if extra_input:
        user_text += f"\n\nEk bilgi:\n{extra_input}"

    user_message = [{"role": "user", "content": user_text}]

    prompt_ids = tokenizer.apply_chat_template(
    user_message,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=False,
    )

    full_ids = tokenizer.apply_chat_template(
    user_message + [{"role": "assistant", "content": answer}],
    tokenize=True,
    add_generation_prompt=False,
    enable_thinking=False,
    return_dict=False,
    )

    labels = full_ids.copy()
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

tokenized_dataset = dataset.map(
    encode_example,
    remove_columns=dataset.column_names,
    load_from_cache_file=False,
)

before_filter = len(tokenized_dataset)
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) <= MAX_LENGTH
)
print(
    f"Eğitime alınan örnek: {len(tokenized_dataset)}; "
    f"uzun olduğu için çıkarılan: {before_filter - len(tokenized_dataset)}"
)

if len(tokenized_dataset) == 0:
    raise ValueError("Filtrelemeden sonra eğitim verisi kalmadı.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

training_args = TrainingArguments(
    output_dir="training_checkpoints_qwen3",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=25,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=0,
    optim="adamw_torch",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Eğitim tamamlandı. Adaptör kaydedildi: {OUTPUT_DIR}")