from .base_llm import BaseLLM
from .sft import test_model, load_rft_dataset, TokenizedDataset, format_example
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import torch

def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example_rft(q, gold, reasoning):
    return {"question": q, "answer": reasoning}

def train_model(
    output_dir: str,
    **kwargs,
):
    # 1. Load RFT data
    trainset = load_rft_dataset()
    # 2. Load base model and tokenizer
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer

    # 3. Add LoRA adapter (increase r if needed)
    r = 16  # You can try 16 or 32 if <50MB
    lora_alpha = 64
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # 4. Prepare tokenized dataset
    tokenized_dataset = TokenizedDataset(tokenizer, trainset, format_example_rft)

    # 5. Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=50,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        label_names=["labels"],
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 7. Train
    trainer.train()

    # 8. Save LoRA adapter
    trainer.save_model(output_dir)

    # 9. Optionally test
    test_model(output_dir)

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})