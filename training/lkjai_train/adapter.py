import json
import os
from pathlib import Path


def train_adapter(paths, settings) -> dict:
    paths.ensure()
    return real_train(paths, settings)


def real_train(paths, settings) -> dict:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as error:
        raise RuntimeError(
            "training dependencies missing; install transformers, peft, bitsandbytes, accelerate, torch"
        ) from error

    dataset_path = Path(os.environ.get("TRAIN_DATASET_PATH", str(paths.fixtures)))
    if not dataset_path.exists():
        raise RuntimeError(f"training dataset not found: {dataset_path}")

    messages_list = load_messages(dataset_path)
    if len(messages_list) < 2:
        raise RuntimeError("real training requires at least 2 dataset rows")

    split = max(1, int(len(messages_list) * (1.0 - settings.eval_ratio)))
    split = min(split, len(messages_list) - 1)
    train_msgs, eval_msgs = messages_list[:split], messages_list[split:]

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def apply_template(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    train_texts = [apply_template(m) for m in train_msgs]
    eval_texts = [apply_template(m) for m in eval_msgs]

    use_4bit = settings.load_in_4bit and torch.cuda.is_available()
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(settings.base_model, **model_kwargs)
    if settings.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = resolve_target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=target_modules,
        ),
    )

    def encode(texts):
        items = []
        for text in texts:
            encoded = tokenizer(text, truncation=True, max_length=settings.sequence_len, padding="max_length")
            encoded["labels"] = encoded["input_ids"][:]
            items.append({k: torch.tensor(v) for k, v in encoded.items()})
        return items

    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, index):
            return self.samples[index]

    train_data = TokenizedDataset(encode(train_texts))
    eval_data = TokenizedDataset(encode(eval_texts))
    out_dir = paths.adapters / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation,
        learning_rate=float(settings.learning_rate),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        max_steps=settings.max_steps if settings.max_steps > 0 else -1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=default_data_collator,
    )
    metrics = trainer.train().metrics
    paths.adapter_final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(paths.adapter_final))
    tokenizer.save_pretrained(str(paths.adapter_final))
    summary = {
        "backend": "transformers-peft-bitsandbytes",
        "checkpoint_dir": str(paths.adapter_final),
        "train_rows": len(train_texts),
        "eval_rows": len(eval_texts),
        "metrics": metrics,
    }
    paths.training_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_messages(path: Path) -> list[list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row.get("messages", []) for row in rows if row.get("messages")]


def resolve_target_modules(model) -> list[str]:
    known_qwen = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    known_gpt2 = ["c_attn", "c_proj", "c_fc"]
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    qwen_hits = [name for name in known_qwen if name in names]
    if qwen_hits:
        return qwen_hits
    gpt2_hits = [name for name in known_gpt2 if name in names]
    if gpt2_hits:
        return gpt2_hits
    generic = sorted(name for name in names if name.endswith("proj") or name.endswith("_proj"))
    if generic:
        return generic[:8]
    raise RuntimeError("unable to infer LoRA target modules from base model")
