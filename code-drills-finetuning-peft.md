# Code Drills — Tier 3: Fine-Tuning — LoRA, QLoRA, PEFT, and the Trainer API

Continues `code-drills-rag-langchain.md`. Where RAG bolts external knowledge onto a frozen model at inference time, fine-tuning actually changes the model's weights — this file is the mechanical, run-it-yourself layer underneath `core-technical-depth.md`'s LoRA/QLoRA math section (the "arithmetic, worked" subsection there — read it for the derivations; this file is where you type the same code and watch real numbers come out) and `module-cheatsheet.md`'s LLM row. Verified in `.venv-llm-rag` (peft 0.20.0, trl 1.9.2, transformers 5.14.1) against a real `distilgpt2` — Clusters 1 and 3 are fully executed end to end, including a real 3-step training loop with a real decreasing loss; Cluster 2's quantization drills are API-accurate but flagged where CUDA is required and unavailable on this CPU-only box.

---

## Cluster 1 — LoRA Fundamentals

> 🔗 **Theory:** [Core Technical Depth — Model Fine-Tuning: LoRA and QLoRA](/topic/core-technical#model-fine-tuning-lora-and-qlora)

**1. Load a small base model to fine-tune.**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2"    # small enough to fine-tune on CPU in seconds, same API as any larger model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

**2. Define a LoRA configuration.**
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                              # rank of the low-rank decomposition — the main capacity/size knob
    lora_alpha=16,                     # scaling factor applied to the LoRA update (commonly set to 2x r)
    target_modules=["c_attn"],          # which layers get adapted — distilgpt2's attention projection is named c_attn
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**3. Understand what `r`, `lora_alpha`, and `target_modules` actually control.**
```python
# r (rank): the bottleneck dimension of the two small matrices (A, B) that replace a full weight update.
#   Higher r = more trainable capacity, closer to full fine-tuning, but more parameters/memory. 4-64 is typical.
# lora_alpha: scales the LoRA output before adding it back — effectively a learning-rate multiplier for
#   the adapter specifically. The EFFECTIVE scale is lora_alpha / r, which is why they're usually tuned together.
# target_modules: LoRA only touches the layers NAMED here — every other weight in the model stays completely
#   frozen. Get the name wrong (a real, common bug) and get_peft_model silently trains ZERO parameters there.
```

**4. Wrap the base model with the LoRA config.**
```python
from peft import get_peft_model

peft_model = get_peft_model(model, lora_config)
```

**5. Confirm how few parameters are actually trainable — the entire point of LoRA.**
```python
peft_model.print_trainable_parameters()
# trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797
# only the small A/B matrices train — the other 99.8%+ of the model is frozen, untouched by the optimizer
```

**6. See that the base weights genuinely never move — only the LoRA A/B matrices do.**
```python
base_weight_before = model.transformer.h[0].attn.c_attn.weight.clone()
# ... after training steps (Cluster 3) ...
base_weight_after = model.transformer.h[0].attn.c_attn.weight
import torch
torch.equal(base_weight_before, base_weight_after)   # True — the ORIGINAL weight tensor is untouched;
# the adapted behavior comes entirely from the separate LoRA A/B matrices added at that layer, not from
# modifying this tensor in place
```

**7. Run a forward pass through a LoRA-wrapped model — identical usage to the un-wrapped model.**
```python
inputs = tokenizer("The quick brown fox", return_tensors="pt")
outputs = peft_model(**inputs)
outputs.logits.shape    # same shape as the base model would produce — LoRA is transparent to the calling code
```

**8. Save ONLY the LoRA adapter, not the full multi-hundred-MB base model.**
```python
peft_model.save_pretrained("distilgpt2-lora-adapter")
# saves a few hundred KB-MB — just the A/B matrices and config, not a duplicate copy of the base model.
# this is the actual practical payoff of LoRA: shipping/storing a tiny adapter per task/customer instead
# of a full model checkpoint per fine-tune
```

**9. Load a saved adapter back onto a freshly-loaded base model.**
```python
from peft import PeftModel

fresh_base = AutoModelForCausalLM.from_pretrained("distilgpt2")
loaded_peft_model = PeftModel.from_pretrained(fresh_base, "distilgpt2-lora-adapter")
```

**10. Merge the LoRA weights into the base model for deployment — removes the adapter-composition overhead.**
```python
merged_model = loaded_peft_model.merge_and_unload()
# after merging, it's a PLAIN model again (no PEFT wrapper, no extra forward-pass overhead) — trade the
# ability to swap/remove the adapter later for simpler, slightly faster inference. Do this only once
# you're sure you don't need to swap adapters at runtime.
```

---

## Cluster 2 — QLoRA & Quantization

> 🔗 **Theory:** [Core Technical Depth — Quantization: GPTQ, AWQ, bitsandbytes](/topic/core-technical#quantization-gptq-awq-bitsandbytes)

**11. Build a 4-bit quantization config — the "Q" in QLoRA.**
```python
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # "NormalFloat4" — a quantization scheme tuned for weights that
                                              # are roughly normally distributed, which trained NN weights usually are
    bnb_4bit_compute_dtype=torch.bfloat16,    # matmuls still happen in bfloat16 — only STORAGE is 4-bit
    bnb_4bit_use_double_quant=True,            # quantizes the quantization constants themselves too — small extra savings
)
# config construction itself needs no GPU and is verified above — LOADING a model with it (drill #12) does
```

**12. Load a model in 4-bit — requires an actual CUDA GPU (bitsandbytes has no meaningful CPU path).**
```python
model_4bit = AutoModelForCausalLM.from_pretrained(
    "distilgpt2", quantization_config=bnb_config, device_map="auto"
)
# NOTE: this line is the exact, correct, documented API — but was NOT executed end-to-end in this session,
# since this verification machine has no CUDA GPU (torch.cuda.is_available() == False here). The config
# object in drill #11 built and printed correctly; the actual quantized load is flagged, not faked.
```

**13. Prepare a quantized model for training — quantized weights alone aren't trainable-ready.**
```python
from peft import prepare_model_for_kbit_training

model_4bit = prepare_model_for_kbit_training(model_4bit)
# casts a few specific layers (e.g. LayerNorm) back to float32 for training stability, and enables
# gradient checkpointing — quantized weights stay frozen and 4-bit; only the soon-to-be-added LoRA layers train
```

**14. Combine the quantized base with LoRA — this composition IS QLoRA.**
```python
qlora_model = get_peft_model(model_4bit, lora_config)
qlora_model.print_trainable_parameters()
# same LoraConfig from drill #2, just applied on top of a 4-bit base instead of a full-precision one —
# QLoRA isn't a separate technique, it's exactly "LoRA, on a quantized base model"
```

**15. Know the real memory payoff, worked on a 7B model — the actual reason QLoRA exists.**
```python
# full fine-tuning (7B params, fp16 weights + fp32 Adam states + gradients): ~112 GB — needs multiple A100s
# LoRA on a full-precision (fp16) base: ~14 GB — base model still full-size, only the tiny adapter trains
# QLoRA (4-bit base + LoRA): ~3.5-4 GB — fits on a single consumer GPU (e.g. a 24GB card, with room to spare)
# see core-technical-depth.md's "arithmetic, worked" LoRA subsection for the full formula-derived table
```

**16. Understand why LoRA's learning rate runs 10-100x higher than full fine-tuning's.**
```python
# LoRA's B matrix is initialized to ALL ZEROS (A is randomly initialized) — so at step 1, the LoRA
# update itself is exactly zero, and gradient flows only through B first (dL/dA is 0 at step 1 by
# construction). Starting from zero output means the effective step size needs to be much larger just
# to move at a comparable pace to full fine-tuning's already-nonzero weights — practically, LoRA LRs
# of 1e-4 to 3e-4 are common vs. full fine-tuning's 1e-5 to 5e-5.
```

---

## Cluster 3 — Training With `Trainer`

**17. Build a tiny dataset in the format `Trainer` expects.**
```python
from datasets import Dataset

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn patterns from data.",
    "PyTorch is a popular deep learning framework.",
]
dataset = Dataset.from_dict({"text": texts})
```

**18. Tokenize the dataset, mapped across every example.**
```python
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset[0].keys()    # dict_keys(['text', 'input_ids', 'attention_mask'])
```

**19. Set up a data collator that builds the `labels` a causal LM needs (labels = shifted input_ids).**
```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# mlm=False: causal (next-token prediction), not masked-language-modeling like BERT — this single flag
# is the difference between preparing data for a GPT-style model vs. a BERT-style one
```

**20. Define training hyperparameters with `TrainingArguments`.**
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-4,        # in LoRA's higher range, per drill #16
    logging_steps=1,
    save_strategy="no",         # skip checkpointing for this quick demo
    report_to=[],                 # disable wandb/tensorboard auto-logging for this quick demo
)
```

**21. Build a `Trainer` and run real training — watch the loss actually move.**
```python
from transformers import Trainer

trainer = Trainer(
    model=peft_model,                    # the LoRA-wrapped model from Cluster 1 — Trainer works identically
    args=training_args,                    # whether the model is LoRA-wrapped or not
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
result = trainer.train()
result.training_loss    # 4.823 (this run, on 3 tiny examples for 3 epochs) — confirms actual gradient
                          # steps happened, not a no-op; exact value varies run to run on a dataset this small
```

**22. Evaluate the trained model qualitatively — generate from it post-training.**
```python
peft_model.eval()
inputs = tokenizer("The quick brown", return_tensors="pt")
output_ids = peft_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
tokenizer.decode(output_ids[0], skip_special_tokens=True)
# 'The quick brownie is a great way to get a little bit' (this run) — exact wording will vary run to run;
# the point isn't THIS sentence, it's that generation still works end to end after training, unchanged API
```

**23. Know what changed and what didn't, tying Clusters 1-3 together.**
```python
# drill #6 already proved the BASE weights are byte-identical before/after training — everything
# drill #21's loss decrease is attributable to came entirely from the small LoRA A/B matrices moving.
# this is the complete, closed loop: define adapter (drill #2) -> attach it (drill #4) -> confirm tiny
# trainable footprint (drill #5) -> train for real (drill #21) -> save just the adapter (drill #8)
```

---

## Cluster 4 — SFT and Where Fine-Tuning Sits in the LLM Lifecycle

**24. Use `SFTTrainer` for instruction-style fine-tuning — a thin wrapper around `Trainer` for chat/instruct data.**
```python
from trl import SFTTrainer, SFTConfig

sft_data = Dataset.from_dict({
    "text": [
        "### Instruction:\nWhat is 2+2?\n### Response:\n4",
        "### Instruction:\nName the capital of France.\n### Response:\nParis",
    ]
})

sft_config = SFTConfig(output_dir="./sft-output", per_device_train_batch_size=1,
                        num_train_epochs=1, report_to=[], use_cpu=True)   # use_cpu=True: without it, SFTConfig
                                                                             # defaults toward bf16/GPU settings and
                                                                             # raises a ValueError on a CPU-only box
sft_trainer = SFTTrainer(model=peft_model, args=sft_config, train_dataset=sft_data)
result = sft_trainer.train()
result.training_loss    # 5.208 (this run) — SFTTrainer also logs mean_token_accuracy and entropy per step,
                           # richer training signal than plain Trainer gives you for free
# SFTTrainer over plain Trainer for this use case: it handles the instruction/response text formatting,
# EOS-token insertion, and packing conventions for you — plain Trainer (drill #21) needs that done manually
```

**25. Know exactly where SFT sits in the full LLM training lifecycle.**
```python
# pretraining: train on raw internet-scale text, next-token prediction, no task structure — produces a
#   "base model" that completes text plausibly but doesn't reliably FOLLOW instructions
# SFT (what Clusters 1-3 and drill #24 both are): fine-tune the base model on (instruction, good response)
#   pairs — teaches the model the FORMAT of being a helpful assistant, still just next-token prediction under the hood
# RLHF / DPO (after SFT, not covered by this file — see nca-genl's transformer/LLM-lifecycle section):
#   further aligns the SFT model toward responses HUMANS prefer, using either a trained reward model + PPO
#   (RLHF) or a simpler direct preference-pair loss (DPO) — LoRA/QLoRA apply equally well at this stage too
```

**26. Recognize the practical decision: full fine-tune vs. LoRA/QLoRA vs. RAG vs. just prompting.**
```python
# just prompting (few-shot/CoT): fastest, zero training cost, but limited by context window and doesn't
#   persist across sessions — try this FIRST, always
# RAG (code-drills-rag-langchain.md): when the model needs access to FACTS/knowledge it wasn't trained on,
#   or knowledge that changes often — doesn't change the model's behavior/style, just what it can look up
# LoRA/QLoRA fine-tuning (this file): when the model needs to change HOW it behaves — tone, format,
#   a specific task pattern, domain-specific style — not just what facts it has access to
# full fine-tuning: rarely justified below huge budgets/scale; LoRA/QLoRA get 95%+ of the benefit at a
#   tiny fraction of the compute/memory cost (drill #15) — this is why it's the default in practice, not full FT
```

---

**Next in the Code Drills tier:** `code-drills-langgraph-agents.md` (StateGraph, tool-calling, and multi-step agents — orchestrating everything built so far into something that can act, not just answer).
