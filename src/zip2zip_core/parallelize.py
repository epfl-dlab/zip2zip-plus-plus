"""Parallelization for the Zip2Zip Llama model.

Uses FSDP for data parallelism. Tensor parallelism is not supported due to
the dynamic hyper-encoder weights.
"""

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger

from zip2zip_core.model import Zip2ZipLlama3Model


_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten.linear.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
}


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def parallelize_zip2zip_llama(
    model: Zip2ZipLlama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply FSDP and activation checkpointing to the zip2zip model."""

    if parallel_dims.tp_enabled:
        logger.warning(
            "Tensor Parallelism is not supported for zip2zip models due to "
            "dynamic hyper-encoder weights. Ignoring TP configuration."
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        for layer_id, transformer_block in model.layers.named_children():
            transformer_block = torch.compile(
                transformer_block, backend=compile_config.backend, fullgraph=True
            )
            model.layers.register_module(layer_id, transformer_block)
        logger.info("Compiling each TransformerBlock with torch.compile")

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)

        param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        pp_enabled = parallel_dims.pp_enabled
        reshard = not pp_enabled

        # Shard embedding
        if model.tok_embeddings is not None:
            fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard)

        # Shard hyper-encoder
        fully_shard(model.hyper_encoder, **fsdp_config, reshard_after_forward=reshard)

        # Shard transformer layers
        for layer_id, transformer_block in model.layers.items():
            fully_shard(transformer_block, **fsdp_config, reshard_after_forward=reshard)

        # Shard output layer
        if model.norm is not None and model.output is not None:
            fully_shard(
                [model.norm, model.output],
                **fsdp_config,
                reshard_after_forward=False,
            )

        fully_shard(model, **fsdp_config)
        disable_fsdp_gradient_division(model)

        logger.info("Applied FSDP to the zip2zip model")

    elif parallel_dims.dp_replicate_enabled:
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        replicate(model, device_mesh=dp_replicate_mesh, bucket_cap_mb=100)
        logger.info("Applied DDP to the zip2zip model")

    return model
