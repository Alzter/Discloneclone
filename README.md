# Discloneclone - Fine-tune LLMs locally on your Discord chat history
Discloneclone is a clone of [Disclone](https://github.com/FlintSH/Disclone) (by FlintSH) that allows
you to create a _clone of yourself_ (or someone else) by fine-tuning an LLM on Discord chat history using your own hardware. It also allows you to run a Discord bot that uses the fine-tuned LLM to respond to incoming messages (WIP).

# What it does
Discloneclone can:
1. Process one or multiple CSV files exported from Discord using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).
2. Create training data from the chat logs, focusing on a specific user.
3. Fine-tune any LLM using the training data on your **local hardware** using [transformers](https://github.com/huggingface/transformers), [trl](https://github.com/huggingface/trl), [peft](https://github.com/huggingface/peft), and [unsloth](https://github.com/unslothai/unsloth).
4. Run a Discord Bot using the fine-tuned LLM to pretend to be you. (WIP)

# Prerequisites
- Python
- CSV files of your Discord conversations exported using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).
- NVIDIA CUDA Toolkit (for fine-tuning on local hardware)

# Installation
1. Clone this repository
```bash
git clone
https://github.com/Alzter/discloneclone
cd discloneclone
```

2. Install the required packages:

    a. Using Pip:
    ```bash
    pip install -r requirements.txt
    ```
    
    b. Using NixPkgs / NixOS:
    ```bash
    nix-shell
    ```
# Usage

1. Export your Discord chat logs as a CSV using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).

2. Preprocess all CSV files in a directory into one dataset:
```python
import json
import parser
from glob import glob
chats = glob("discord-chat-folder/*.csv") # Obtain all chat logs in folder
dataset = parser.create_dataset(chats, "your-username") # Create training dataset from chat logs
with open("data_combined.json", "w") as f:
    json.dump(data, f, indent=2) # Export to JSON file
```

3. Fine-tune an LLM on the data locally
```bash
python src/utils/run_finetune.py finetune_args.json
```

<details>
  <summary>finetune_args.json</summary>
  
  ```json
  {
    "dataset" : "data_combined.json",
    "test_size" : 0,
    "model_name_or_path" : "microsoft/Phi-4-mini-instruct",
    "cuda_devices" : "0",
    "use_4bit_quantization" : true,
    "bnb_4bit_quant_type" : "nf4",
    "bnb_4bit_compute_dtype" : "float16",
    "use_nested_quant" : true,
    "use_reentrant" : false,
    "attn_implementation" : "sdpa",
    "output_dir" : "models/MyModel",
    "use_peft_lora" : true,
    "lora_target_modules" : "all-linear",
    "lora_r" : 6,
    "lora_alpha" : 8,
    "lora_dropout" : 0.05,
    "max_seq_length" : 128,
    "num_train_epochs" : 1,
    "learning_rate" : 2e-4,
    "optim" : "adamw_torch_fused",
    "warmup_ratio" : 0.03,
    "lr_scheduler_type" : "constant",
    "packing" : true,
    "logging_steps" : 10,
    "logging_dir" : "./logs",
    "report_to" : "none",
    "gradient_checkpointing" : true,
    "gradient_accumulation_steps" : 1,
    "per_device_train_batch_size" : 16,
    "auto_find_batch_size" : true
  }
  ```
</details>

4. Run a Discord bot using the fine-tuned LLM

```bash
python src/main.py
```

# Documentation
- All code is documented with in-line DocStrings.

# Credits
- FlintSH for the original Disclone, of which this project is inspired by
