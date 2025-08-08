# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass

import datasets
import torch
import torch.distributed
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig
from transformers.trainer_utils import SaveStrategy
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTTrainer

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
from dotenv import load_dotenv
from umap import UMAP
load_dotenv()

BASE_DIR = os.getenv('BASE_DIR')
HF_TOKEN = os.getenv('HF_TOKEN')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_model(
        vocab_size: int,
        model_path: str,
):
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        head_dim=128,
        hidden_size=128,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        tie_word_embeddings=True
    )
    model = LlamaForCausalLM(llama_config)

    #E = reduced_embeddings(model_path=model_path, hidden_size=llama_config.hidden_size)
    #E = torch.rand(model.get_input_embeddings().weight.size(), dtype=torch.float16)
    #model.get_input_embeddings().weight.data.copy_(E)
    #model.get_output_embeddings().weight.data.copy_(E)

    #model.set_input_embeddings(torch.nn.Embedding.from_pretrained(E, freeze=True))
    #model.set_output_embeddings(torch.nn.Linear(llama_config.hidden_size, llama_config.vocab_size, bias=False))

    return model


@dataclass
class DataArguments:
    dataset_name: str | None = "Open-Orca/OpenOrca"
    data_type: str | None = "HF"  # HF or JSON
    total_batch_size: int = 64 #add this here since training args cannot handle it

@dataclass
class ModelArguments:
    teacher_name_or_path: str | None = None
    student_name_or_path: str | None = None
    single_model: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    do_train: bool = True
    do_eval: bool = True
    save_strategy: str = "steps"
    save_steps : int = 1500
    save_total_limit : int = 1
    max_length: int = 1024
    optim: str = "adamw_torch"
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    dataloader_drop_last: bool = True
    dataset_num_proc: int = 8
    #dataset_batch_size: int = 500
    bf16: bool = True
    tf32: bool = True


def llama_test_func(sample):
    """Formatting function for Llama-style datasets."""
    return sample["text"]

def save_model_old(trainer: transformers.Trainer):
    """Dumps model and ModelOpt states to disk."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
        cpu_state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            output_dir = trainer.args.output_dir
            trainer._save(output_dir, state_dict=cpu_state_dict)
            # ModelOpt state
            logger.info(f"Saving modelopt state to {output_dir}")
            torch.save(mto.modelopt_state(model), f"{output_dir}/modelopt_state.pt")


def save_model(trainer: transformers.Trainer):
    """Dumps model and ModelOpt states to disk."""
    if not trainer.args.should_save:
        return

    model = trainer.accelerator.unwrap_model(trainer.model)
    output_dir = trainer.args.output_dir

    # Let the trainer handle FSDP state dict collection
    trainer.save_model(output_dir)

    # Save ModelOpt state
    logger.info(f"Saving modelopt state to {output_dir}")
    torch.save(mto.modelopt_state(model), f"{output_dir}/modelopt_state.pt")


def reduced_embeddings(model_path, hidden_size=128):
    E = _teacher_factory(model_path).get_input_embeddings().weight
    umap_model = UMAP(
        n_components=hidden_size,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        verbose=True,
    )

    reduced_embeddings = umap_model.fit_transform(E.detach().cpu().numpy())
    reduced_embeddings = torch.tensor(reduced_embeddings, dtype=E.dtype, device=E.device)
    return reduced_embeddings


class KDSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, *args, **kwargs):
        if not model.training:
            _compute_loss_func = self.compute_loss_func
            self.compute_loss_func = None

        loss = super().compute_loss(model, inputs, *args, **kwargs)

        if not model.training:
            self.compute_loss_func = _compute_loss_func

        return loss

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] =  round(tr_loss_scalar /  (self.args.gradient_accumulation_steps*(self.state.global_step - self._globalstep_last_logged)), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def _teacher_factory(model_name_or_path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # enable 4-bit quantization
        bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
    )

    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=PartialState().process_index,
        quantization_config=quantization_config,
        token=HF_TOKEN,
    )


class LMLogitsLoss(mtd.LogitsDistillationLoss):
    def forward(self, out_student: CausalLMOutputWithPast, out_teacher: CausalLMOutputWithPast):
        return super().forward(out_student.logits, out_teacher.logits)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    #torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    # Set total batch size across all ranks to equal 64
    # With this:
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    total_batch_size = data_args.total_batch_size
    num_accum_steps = total_batch_size / (
        training_args.per_device_train_batch_size * world_size
    )
    if not num_accum_steps.is_integer():
        raise ValueError(
            f"`per_device_train_batch_size` * `world_size` must be a factor of {total_batch_size}"
        )
    training_args.gradient_accumulation_steps = int(num_accum_steps)
    logger.info(
        f"Using {int(num_accum_steps)} grad accumulation steps for effective batchsize of {total_batch_size}."
    )

    logger.info("Loading dataset...")
    if data_args.data_type == "HF":
        dset = datasets.load_dataset(data_args.dataset_name, split="train")
        dset_splits = dset.train_test_split(train_size=25600, test_size=1700, seed=420)
    elif data_args.data_type == "JSON":
        dset = datasets.load_dataset(
            "json",
            data_files=data_args.dataset_name,
            split="train",
        )
        dset_splits = dset.train_test_split(train_size=None, test_size=1700, seed=420)
    else:
        raise ValueError(f"Unsupported dataset type: {data_args.dataset_type}")

    dset_train, dset_eval = dset_splits["train"], dset_splits["test"]
    logger.info("Dataset loaded.")

    logger.info("Loading tokenizer...")
    model_path = model_args.teacher_name_or_path or model_args.student_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")

    if model_args.single_model:
        logger.info("Loading single model only...")
        model = _teacher_factory(model_path)
        logger.info("Model loaded.")
    else:
        logger.info("Loading student model...")
        student_model = setup_model(
            vocab_size=len(tokenizer),
            model_path=model_path,
        )

        # student_model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_args.student_name_or_path,
        #     device_map=PartialState().process_index,
        # )
        logger.info("Student loaded.")

        logger.info("Loading teacher model and converting to Distillation model...")
        kd_config = {
            "teacher_model": (
                _teacher_factory,
                (model_args.teacher_name_or_path,),
                {},
            ),
            "criterion": LMLogitsLoss(),
            "expose_minimal_state_dict": False,  # FSDP forces us to disable this
        }
        model = mtd.convert(student_model, mode=[("kd_loss", kd_config)])
        logger.info("Models converted.")

    # Fix problematic settings that logger.info excessive warnings
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Trainer
    trainer_cls = SFTTrainer if model_args.single_model else KDSFTTrainer
    trainer = trainer_cls(
        model,
        training_args,
        train_dataset=dset_train,
        eval_dataset=dset_eval,
        formatting_func=llama_test_func,
        processing_class=tokenizer,
    )
    if isinstance(trainer, KDSFTTrainer):
        # Use our distillation aggregate loss
        trainer.compute_loss_func = lambda *a, **kw: model.compute_kd_loss()

    # Do training
    if training_args.do_train:
        # Load checkpoint
        checkpoint = training_args.resume_from_checkpoint
        if checkpoint and not model_args.single_model:
            # ModelOpt state
            modelopt_state_path = os.path.join(checkpoint, "modelopt_state.pt")
            print(modelopt_state_path)
            if not os.path.isfile(modelopt_state_path):
                raise FileNotFoundError("`modelopt_state.pt` not found with checkpoint.")
            logger.info(f"Loading modelopt state from {modelopt_state_path}")
            modelopt_state = torch.load(modelopt_state_path, weights_only=False)
            mto.restore_from_modelopt_state(model, modelopt_state)

        logger.info("Beginning training...")
        trainer.train(resume_from_checkpoint=checkpoint)
        logger.info("Training done.")

    # Do evaluation
    if training_args.do_eval:
        logger.info("Evaluating...")
        eval_results = trainer.evaluate()
        logger.info(eval_results)
        logger.info("Evalutation complete.")

    # Save checkpoint
    logger.info("Saving checkpoint...")
    save_model(trainer)
    logger.info("Checkpoing saved.")


if __name__ == "__main__":
    train()
