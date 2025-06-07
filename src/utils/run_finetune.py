import os
import sys

from transformers import HfArgumentParser
from trl import SFTConfig
from utils import LocalModelArguments, DatasetArguments
from datasets import DatasetDict

import torch
import numpy as np

# needed for loading models
#torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType])

#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def main(local_model_args : LocalModelArguments, data_args : DatasetArguments, training_args : SFTConfig):
    from transformers import set_seed
    from trl import SFTTrainer
    from utils import LocalPLM
    
    if not local_model_args.model_name_or_path:
        raise ValueError("Argument required: model_name_or_path")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load training/evaluation dataset
    import datasets
    dataset = datasets.load_dataset("json", data_files=data_args.dataset) 
    if type(dataset) is DatasetDict: dataset = dataset['train']
    if data_args.test_size:
        dataset = dataset.train_test_split(data_args.test_size)
    
    if data_args.num_shards > 0:
        dataset = dataset.shard(data_args.num_shards, data_args.shard_index)

    dataset = dataset.shuffle(seed=training_args.seed) # Randomly shuffle data
    
    # import preprocess as pre
    # dataset, label_names = pre.load_dataset(
    #     data_args.dataset,
    #     data_args.text_columns,
    #     data_args.label_columns,
    #     test_size = data_args.test_size,
    #     ratio = data_args.ratio,
    #     size = data_args.size
    # )

    # train_dataset, eval_dataset = dataset['train'], dataset['test']

    #training_args.dataset_kwargs = {
    #    "append_concat_token": data_args.append_concat_token,
    #    "add_special_tokens": data_args.add_special_tokens,
    #}

    # datasets
    #train_dataset, eval_dataset = create_datasets(
    #    tokenizer,
    #    data_args,
    #    training_args,
    #    apply_chat_template=local_model_args.chat_template_format != "none",
    #)
    
    # model
    model = LocalPLM(local_model_args, training_args=training_args)
    
    import subprocess
    subprocess.run(["nvidia-smi"])
    
    # Enable gradient checkpointing (saves memory)
    model.model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not local_model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": local_model_args.use_reentrant}
    
    # Finetune the model
    trainer, history = model.finetune(
        train_dataset=dataset['train'] if type(dataset) is DatasetDict else dataset,
        eval_dataset=dataset['test'] if type(dataset) is DatasetDict else None,
        sft_config=training_args,
        checkpoint=training_args.resume_from_checkpoint
    )
    
    # Save training history
    model.save_training_history(history, training_args.output_dir)

    # trainer
    # trainer = SFTTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     args=training_args,
    #     train_dataset=dataset['train'],
    #     eval_dataset=dataset['test'],
    #     peft_config=peft_config,
    # )

    # # trainer.accelerator.print(f"{trainer.model}")
    # # if hasattr(trainer.model, "print_trainable_parameters"):
    # #     trainer.model.print_trainable_parameters()

    # # train
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint)

    # # saving final model
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()


if __name__ == "__main__":

    parser = HfArgumentParser((LocalModelArguments, DatasetArguments, SFTConfig))
    
    # If we pass only one argument to the script and it's
    # the path to a json file, let's parse it to get our arguments.

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        # We must assign CUDA_VISIBLE_DEVICES here
        # before transformers and torch are imported
        file = os.path.abspath(sys.argv[1])
        import json
        with open(file) as f:
            data = json.load(f)
            f.close()
        if data.get("cuda_devices"):
            os.environ["CUDA_VISIBLE_DEVICES"] = data["cuda_devices"]
        del data

        args = parser.parse_json_file(json_file=file)
    
    else:
        # We must assign CUDA_VISIBLE_DEVICES here
        # before transformers and torch are imported
        args, _ = parser.parse_known_args()
        if args.cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

        args = parser.parse_args_into_dataclasses()
    
    local_model_args, data_args, training_args = args

    main(local_model_args, data_args, training_args)
