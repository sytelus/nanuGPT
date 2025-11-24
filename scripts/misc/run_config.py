import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any, List, Optional

@dataclass
class RunConfig:
    @dataclass
    class Environment:
        experiment_name: str = field(metadata={"help": "Name of the experiment."})
        experiment_uuid: str = field(
            metadata={"help": "UUID of the experiment."}
        )


    @dataclass
    class Logging:
        log_params_norm: bool = field(
            metadata={"help": "Calculate and log the norm of parameters if set to True."}
        )
        log_per_param_stats: bool = field(

            metadata={"help": "Calculate and log statistics for each parameter if set to True."},
        )
        params_norm_log_interval: int = field(

            metadata={"help": "Specify the interval (in steps) at which to log the parameters norm."},
        )
        sync_timer: bool = field(

            metadata={
                "help": "Enable synchronization in Timers for more accurate timing reports. May cause slowdowns in highly optimized configurations."
            },
        )
        cuda_event_timer: bool = field(

            metadata={
                "help": "Use CUDA events for Timers if set to True. Provides accuracy with minimal performance impact."
            },
        )
        log_stats_by_group: bool = field(

            metadata={
                "help": "Calculate and log statistics for parameter/gradient groups if set to True."
            },
        )
        log_num_zeros_in_grad: bool = field(

            metadata={"help": "Calculate and log the number of zeros in gradients if set to True."},
        )
        log_fp8_stats: bool = field(

            metadata={"help": "Log FP8-related statistics, such as scaling factors, if set to True."},
        )
        log_moe_stats: bool = field(

            metadata={"help": "Log MoE-related statistics, such as routing states, if set to True."},
        )
        table_logging: bool = field(
            metadata={"help": "Enable logging to Azure Table if set to True."}
        )
        prometheus_metric_logging: bool = field(
            metadata={"help": "Enable metric logging to Prometheus if set to True."}
        )
        wandb_logging: bool = field(

            metadata={"help": "Enable logging to Weights & Biases (wandb) if set to True."},
        )
        wandb_rank_0: bool = field(

            metadata={"help": "Log only on Rank 0 if set to True when using Weights & Biases."},
        )
        verbose: bool = field(
            metadata={"help": "Print detailed model weight information if set to True."}
        )
        verbose_level: int = field(

            metadata={
                "help": "Set the verbosity level for logging. Higher levels print more detailed information."
            },
        )
        table_log_name: Optional[str] = field(
            metadata={"help": "Specify the name for logging to Azure Table."}
        )
        wandb_name: Optional[str] = field(
            metadata={"help": "Specify the name for logging to Weights & Biases (wandb)."}
        )
        wandb_project: str = field(

            metadata={"help": "Specify the project name for Weights & Biases (wandb)."},
        )
        log_tensors_file: Optional[str] = field(

            metadata={
                "help": "Log tensors to a specified file in NPZ format. Use '%(rank)s' to include the `args.rank` in the filename."
            },
        )
        log_refinery_tag: Optional[str] = field(
            metadata={"help": "Specify a file tag for logging job status."}
        )
        nsys_profile_steps: int = field(

            metadata={
                "help": "Specify the number of steps to profile using NVIDIA Nsight Systems (nsys)."
            },
        )
        mem_profile_steps: int = field(

            metadata={"help": "Specify the number of steps to profile using PyTorch memory profiler."},
        )
        torch_profile_steps: int = field(

            metadata={"help": "Specify the number of steps to profile using PyTorch profiler."},
        )
        log_full_loss: bool = field(

            metadata={"help": "Log the full (long) representation of the loss if set to True."},
        )
        log_all_ranks: bool = field(
            metadata={"help": "Enable distributed logging on every GPU if set to True."}
        )
        log_ep0_tp0_ranks: bool = field(

            metadata={
                "help": "Enable distributed logging on GPUs with EP rank 0 and TP rank 0 if set to True."
            },
        )
        log_pp_ranks: bool = field(

            metadata={
                "help": "Enable distributed logging on GPUs in the last TP (Tensor Parallel) and DP (Data Parallel) ranks, but with different PP (Pipeline Parallel) ranks if set to True."
            },
        )


    @dataclass
    class Data:
        train_data_path: Optional[str] = field(
            metadata={"help": "File path to the preprocessed training dataset."}
        )
        eval_tasks: List[str] = field(
            metadata={"help": "List of evaluation tasks to be performed."}
        )
        data_tmpdir: str = field(

            metadata={"help": "Temporary directory for storing preprocessed data."},
        )
        seq_length: Optional[int] = field(
            metadata={"help": "Maximum sequence length for input processing."}
        )
        clamp_seq_len: Optional[int] = field(
            metadata={"help": "Fixed sequence length for evaluation tasks."}
        )
        encoder_seq_length: Optional[int] = field(

            metadata={
                "help": "Maximum sequence length for the encoder. Should be specified separately from --seq-length."
            },
        )
        decoder_seq_length: Optional[int] = field(
            metadata={"help": "Maximum sequence length for the decoder."}
        )
        retriever_seq_length: int = field(

            metadata={
                "help": "Maximum sequence length for the retriever model in the biencoder setup."
            },
        )
        num_workers: int = field(
            metadata={"help": "Number of worker threads for data loading."}
        )
        tokenizer: str = field( metadata={"help": "Name of the tokenizer to be used."})
        reset_position_ids: bool = field(

            metadata={"help": "Reset position IDs after encountering an end-of-document token."},
        )
        reset_attention_mask: bool = field(

            metadata={"help": "Reset self-attention mask after encountering an end-of-document token."},
        )
        eod_mask_loss: bool = field(
            metadata={"help": "Apply loss masking for end-of-document tokens."}
        )
        nccl_replicate_data: bool = field(

            metadata={
                "help": "Enable NCCL broadcasting to share preprocessed data shards across nodes."
            },
        )
        preload_training_data: bool = field(

            metadata={
                "help": "Load the entire training dataset into memory before starting training. This is required when using --nccl-replicate-data."
            },
        )
        train_validation_iters: Optional[int] = field(

            metadata={
                "help": "Number of iterations to run over the validation set during training. If not specified, the full validation set will be used."
            },
        )


    @dataclass
    class Distributed:
        tensor_model_parallel_size: int = field(
            metadata={"help": "Degree of parallelism for tensor model operations."}
        )
        pipeline_model_parallel_size: int = field(
            metadata={"help": "Degree of parallelism for pipeline model operations."}
        )
        expert_parallel_size: int = field(
            metadata={"help": "Degree of parallelism for expert models."}
        )
        danger_override_expert_parallel_optimizer: bool = field(

            metadata={
                "help": "Override the expert parallelism requirement for the distributed Adam optimizer. WARNING: This may produce incorrect results."
            },
        )
        pipeline_model_parallel_split_rank: Optional[int] = field(

            metadata={
                "help": "Rank at which to split the encoder and decoder for pipeline parallelism."
            },
        )
        num_layers_per_virtual_pipeline_stage: Optional[str] = field(

            metadata={
                "help": (
                    "Specify the number of layers per virtual pipeline stage. Acceptable formats are: "
                    '(1) A single integer "A" for uniform layers per stage, '
                    '(2) Two integers "A_B" for different layers in different parts of the model, '
                    '(3) A detailed format "AxX_BxY_CxZ_..." where "AxX" means "X" stages each with "A" layers.'
                )
            },
        )
        num_layers_per_pipeline_stage: Optional[str] = field(

            metadata={
                "help": (
                    'Specify the number of layers per pipeline stage in the format "W_X_Y_Z", where each number '
                    'represents the layers in that stage. Supports an arbitrary number of stages. For example, "X_Y" for two stages.'
                )
            },
        )
        apex_overlap_p2p: bool = field(

            metadata={
                "help": "Enable overlapping point-to-point communication with computation using APEX. Applies to interleaved pipeline parallelism."
            },
        )
        distributed_backend: str = field(
            metadata={"help": "Distributed backend to use for training."}
        )
        distributed_store: str = field(

            metadata={"help": "Key-value store to use for rendezvous in init_process_group."},
        )
        pg_timeout_minutes: float = field(
            metadata={"help": "Timeout for process group initialization in minutes."}
        )
        pg_init_parallel: bool = field(

            metadata={
                "help": "Initialize all process groups simultaneously to speed up initialization."
            },
        )
        DDP_impl: str = field(
            metadata={"help": "Implementation of DistributedDataParallel to use."}
        )
        local_DDP_allreduce_bucket_count: int = field(

            metadata={
                "help": "Number of buckets to split model weight gradients for asynchronous all-reduce in local DDP."
            },
        )
        use_contiguous_buffers_in_local_ddp: bool = field(
            metadata={"help": "Use contiguous buffers in local DDP for efficiency."}
        )
        tp_overlap: bool = field(

            metadata={
                "help": "Enable Tensor Parallel communication overlap. Effective only with certain transformer implementations (e.g., 'transformer_engine' or 'mixed_v2')."
            },
        )
        scatter_gather_tensors_in_pipeline: bool = field(

            metadata={
                "help": "Optimize tensor communication in the pipeline using scatter/gather operations."
            },
        )
        use_ring_exchange_p2p: bool = field(

            metadata={
                "help": (
                    "Use custom-built ring exchange for point-to-point communications if enabled. "
                    "Requires a custom-built image supporting ring-exchange p2p."
                )
            },
        )
        lazy_mpu_init: bool = field(

            metadata={
                "help": (
                    "If True, defer DDP initialization in initialize_megatron() and return a function to complete it later. "
                    "Also enables --use-cpu-initialization. Intended for use with an external DDP manager."
                )
            },
        )
        use_cpu_initialization: bool = field(

            metadata={"help": "Initialize affine parallel weights on the CPU if enabled."},
        )
        empty_unused_memory_level: int = field(

            metadata={
                "help": (
                    "Level of memory cleanup to perform after each iteration (training and evaluation) "
                    "to reduce GPU memory fragmentation. Options are: 0=off, 1=moderate cleanup, 2=aggressive cleanup."
                )
            },
        )
        standalone_embedding_stage: bool = field(

            metadata={
                "help": (
                    "If set to True, the input embedding layer is placed on its own pipeline stage, separate from transformer layers. "
                    "Currently, for T5 models, this affects only the encoder embedding."
                )
            },
        )

        def __post_init__(self) -> None:
            DISTRIBUTED_BACKEND = ["Nccl", "gloo"]
            DISTRIBUTED_STORE = ["default", "redis"]
            DDP_IMPL = ["local", "torch"]
            EMPTY_UNUSED_MEMORY_LEVEL = [0, 1, 2]

            if self.distributed_backend not in DISTRIBUTED_BACKEND:
                raise ValueError(
                    f"`distributed_backend` should be one of {DISTRIBUTED_BACKEND}, got {self.distributed_backend}"
                )
            if self.distributed_store not in DISTRIBUTED_STORE:
                raise ValueError(
                    f"`distributed_store` should be one of {DISTRIBUTED_STORE}, got {self.distributed_store}"
                )
            if self.DDP_impl not in DDP_IMPL:
                raise ValueError(f"`DDP_impl` should be one of {DDP_IMPL}, got {self.DDP_impl}")
            if self.empty_unused_memory_level not in EMPTY_UNUSED_MEMORY_LEVEL:
                raise ValueError(
                    f"`empty_unused_memory_level` should be one of {EMPTY_UNUSED_MEMORY_LEVEL}, got {self.empty_unused_memory_level}"
                )


    @dataclass
    class MixedPrecision:
        fp16: bool = field(

            metadata={
                "help": "Enable model training and inference in 16-bit floating point (fp16) mode."
            },
        )
        bf16: bool = field(

            metadata={"help": "Enable model training and inference in bfloat16 (bf16) mode."},
        )
        loss_scale: Optional[float] = field(

            metadata={
                "help": (
                    "Static loss scaling factor, typically a power of 2, which can help improve convergence in fp16 mode. "
                    "If set to None, dynamic loss scaling will be used instead."
                )
            },
        )
        initial_loss_scale: float = field(
            metadata={"help": "Initial loss scale value for dynamic loss scaling."}
        )
        min_loss_scale: float = field(

            metadata={"help": "Minimum allowable loss scale value during dynamic loss scaling."},
        )
        loss_scale_window: float = field(

            metadata={"help": "Window size (in iterations) for adjusting the dynamic loss scale."},
        )
        hysteresis: int = field(

            metadata={
                "help": "Hysteresis for dynamic loss scaling, controlling the number of steps to wait before adjusting the loss scale."
            },
        )
        fp32_residual_connection: bool = field(

            metadata={
                "help": "Perform residual connections in 32-bit floating point (fp32) to improve numerical stability."
            },
        )
        query_key_layer_scaling: bool = field(

            metadata={
                "help": "Scale the dot product of query and key vectors by 1 / layer-number for improved stability."
            },
        )
        attention_softmax_in_fp32: bool = field(

            metadata={
                "help": (
                    "Perform attention masking and softmax operations in 32-bit floating point (fp32) mode. "
                    "This flag is ignored if query-key layer scaling is enabled."
                )
            },
        )
        accumulate_allreduce_grads_in_fp32: bool = field(

            metadata={
                "help": "Accumulate gradients and perform all-reduce operations in 32-bit floating point (fp32) mode."
            },
        )
        fp16_lm_cross_entropy: bool = field(

            metadata={
                "help": "Compute the unreduced cross-entropy loss for the language model head in 16-bit floating point (fp16) mode."
            },
        )


    @dataclass
    class TransformerEngine:
        fp8_e4m3: bool = field(

            metadata={"help": "Enable the E4M3 format for Transformer layers in FP8 precision."},
        )
        fp8_hybrid: bool = field(

            metadata={
                "help": "Enable the hybrid FP8 format for Transformer layers, combining different FP8 formats for optimal performance."
            },
        )
        fp8_wgrad: bool = field(

            metadata={
                "help": "Execute weight gradient computations in higher precision, even when using FP8."
            },
        )
        fp8_margin: int = field(

            metadata={
                "help": "Margin for scaling in FP8 operations. Adjust to balance range and precision."
            },
        )
        fp8_interval: int = field(

            metadata={
                "help": "Update interval for FP8 scaling factors, defined in terms of iterations."
            },
        )
        transformer_impl: str = field(
            metadata={"help": "Specify which Transformer implementation to use."}
        )
        fp8_amax_history_len: int = field(

            metadata={
                "help": "Number of steps for which the amax (absolute max) history is recorded per tensor for FP8 scaling."
            },
        )
        fp8_amax_compute_algo: str = field(
            metadata={"help": "Algorithm to compute amax from history."}
        )
        custom_amax: bool = field(

            metadata={
                "help": "Use a custom algorithm for amax computation. This is effective only when using the 'mixed_v2' implementation."
            },
        )
        reduce_amax_across_dp: bool = field(

            metadata={
                "help": "Reduce the amax values across data parallel groups to ensure consistent scaling factors."
            },
        )

        def __post_init__(self) -> None:
            TRANSFORMER_IMPL = ["local", "transformer_engine", "mixed", "mixed_v2"]
            FP8_AMAX_COMPUTE_ALGO = ["most_recent", "max"]

            if self.transformer_impl not in TRANSFORMER_IMPL:
                raise ValueError(
                    f"`transformer_impl` should be one of {TRANSFORMER_IMPL}, got {self.transformer_impl}"
                )
            if self.fp8_amax_compute_algo not in FP8_AMAX_COMPUTE_ALGO:
                raise ValueError(
                    f"`fp8_amax_compute_algo` should be one of {FP8_AMAX_COMPUTE_ALGO}, got {self.fp8_amax_compute_algo}"
                )


    @dataclass
    class NetworkSize:
        num_layers: Optional[int] = field(
            metadata={"help": "Number of transformer layers in the model."}
        )
        hidden_size: Optional[int] = field(
            metadata={"help": "Dimension of the hidden layers in the transformer."}
        )
        ffn_hidden_size: Optional[int] = field(

            metadata={
                "help": (
                    "Dimension of the hidden layers in the Feed-Forward Network (FFN) within the transformer. "
                    "If not provided, it defaults to 4 times the hidden size."
                )
            },
        )
        num_attention_heads: Optional[int] = field(

            metadata={"help": "Number of attention heads in the multi-head attention mechanism."},
        )
        kv_channels: Optional[int] = field(

            metadata={
                "help": (
                    "Dimension of the projection weights in the multi-head attention mechanism. "
                    "Defaults to hidden_size divided by num_attention_heads if not specified."
                )
            },
        )
        max_position_embeddings: Optional[int] = field(

            metadata={
                "help": "Maximum number of position embeddings. Defines the range of positions that the model can attend to."
            },
        )
        pos_encoding_type: str = field(
            metadata={"help": "Type of position encoding used."}
        )
        attention_backend: str = field(
            metadata={"help": "Backend implementation for the attention mechanism."}
        )
        attention_type: str = field( metadata={"help": "Type of attention mechanism."})
        num_query_groups: int = field(

            metadata={
                "help": "Number of groups for grouped query attention. If set to zero, grouped query attention (GQA) is disabled."
            },
        )
        vocab_size: Optional[int] = field(

            metadata={"help": "Size of the vocabulary. Required for preprocessed datasets."},
        )
        make_vocab_size_divisible_by: int = field(

            metadata={
                "help": "Pad the vocabulary size to be divisible by this value for computational efficiency."
            },
        )
        layernorm_epsilon: float = field(

            metadata={"help": "Epsilon value used in layer normalization to prevent division by zero."},
        )
        norm_type: str = field( metadata={"help": "Normalization method to use."})
        use_extra_norm: bool = field(

            metadata={
                "help": "Insert an extra normalization layer at the end of each residual block if set to True."
            },
        )
        attn_norm_pattern: str = field(

            metadata={
                "help": (
                    "Pattern for applying normalization in the attention mechanism. "
                    "For example, 'QK' means normalize queries and keys, 'QKP' means normalize queries, keys, and projections."
                )
            },
        )
        use_upstream_linear_layer: bool = field(

            metadata={
                "help": "Use the linear layer implementation from the upstream library for the final projection if set to True."
            },
        )
        apply_residual_connection_post_layernorm: bool = field(

            metadata={
                "help": "Apply the residual connection after layer normalization, mimicking the original BERT architecture if set to True."
            },
        )
        openai_gelu: bool = field(

            metadata={
                "help": "Use OpenAI's GeLU activation function implementation for backward compatibility if set to True."
            },
        )
        onnx_safe: bool = field(

            metadata={
                "help": "Enable workarounds for known issues with the Torch ONNX exporter if set to True."
            },
        )
        qkv_bias: bool = field(

            metadata={
                "help": "Include a bias term in the query, key, and value projections if set to True."
            },
        )

        def __post_init__(self) -> None:
            POS_ENCODING_TYPE = [
                "learned",
                "relative",
                "rope",
                "alibi",
                "absolute",
                "relative_one_q",
                "relative_one_q_no_proj",
            ]
            ATTENTION_BACKEND = ["megatron", "flash_triton", "imha", "imha_nondet", "flash_oss"]
            ATTENTION_TYPE = ["self", "self_multi_q"]
            NORM_TYPE = ["layernorm", "rmsnorm"]

            if self.pos_encoding_type not in POS_ENCODING_TYPE:
                raise ValueError(
                    f"`pos_encoding_type` should be one of {POS_ENCODING_TYPE}, got {self.pos_encoding_type}"
                )
            if self.attention_backend not in ATTENTION_BACKEND:
                raise ValueError(
                    f"`attention_backend` should be one of {ATTENTION_BACKEND}, got {self.attention_backend}"
                )
            if self.attention_type not in ATTENTION_TYPE:
                raise ValueError(
                    f"`attention_type` should be one of {ATTENTION_TYPE}, got {self.attention_type}"
                )
            if self.norm_type not in NORM_TYPE:
                raise ValueError(f"`norm_type` should be one of {NORM_TYPE}, got {self.norm_type}")


    @dataclass
    class Initialization:
        seed: int = field(

            metadata={
                "help": "Random seed for Python, NumPy, PyTorch, and CUDA to ensure reproducibility."
            },
        )
        data_parallel_random_init: bool = field(

            metadata={
                "help": "Enable random initialization of parameters across data parallel ranks instead of synchronized initialization."
            },
        )
        init_method_std: float = field(

            metadata={
                "help": "Standard deviation for the zero-mean normal distribution used in weight initialization."
            },
        )
        init_method_xavier_uniform: bool = field(

            metadata={"help": "Enable Xavier uniform initialization for model parameters."},
        )
        wd_improved: bool = field(

            metadata={
                "help": "Enable improved weight decay for better regularization during training."
            },
        )
        mup: bool = field(

            metadata={
                "help": (
                    "Use Maximal Update Parameterization (MuP) for better hyperparameter transferability. "
                    "This implementation follows the version described in Table 8 of the MuTransfer paper (https://arxiv.org/pdf/2203.03466.pdf). "
                    "MuP is designed to ensure that hyperparameters such as learning rate and beta2 found in smaller models transfer well to larger models."
                )
            },
        )
        mup_base_hidden_size: int = field(

            metadata={
                "help": "Base hidden size dimension used for transfer in Maximal Update Parameterization."
            },
        )
        mup_base_ffn_multiple: int = field(

            metadata={
                "help": "Ratio of the Feed-Forward Network (FFN) hidden size to the base hidden size in the base model for MuP."
            },
        )
        mup_base_head_size: int = field(

            metadata={
                "help": "Base dimension of the attention heads used for transfer in Maximal Update Parameterization."
            },
        )


    @dataclass
    class LearningRate:
        lr: Optional[float] = field(

            metadata={
                "help": (
                    "Initial learning rate at the beginning of training. The learning rate will change over time "
                    "based on the specified decay style and warmup schedule."
                )
            },
        )
        lr_decay_style: str = field(
            metadata={"help": "Style of learning rate decay."}
        )
        lr_decay_iters: Optional[int] = field(

            metadata={
                "help": (
                    "Number of iterations over which to decay the learning rate. "
                    "If not specified, defaults to the total number of training iterations (`--train-iters`)."
                )
            },
        )
        lr_decay_samples: Optional[int] = field(

            metadata={
                "help": (
                    "Number of samples over which to decay the learning rate. "
                    "If not specified, defaults to the total number of training samples (`--train-samples`)."
                )
            },
        )
        lr_decay_at_percent: int = field(

            metadata={
                "help": (
                    "Percentage of the total training process at which to start decaying the learning rate. "
                    "`--lr-decay-iters` determines the duration of the decay."
                )
            },
        )
        lr_warmup_fraction: Optional[float] = field(

            metadata={
                "help": (
                    "Fraction of total warmup iterations or samples to use for learning rate warmup. "
                    "This should be a float value between 0 and 1."
                )
            },
        )
        lr_warmup_iters: int = field(

            metadata={"help": "Number of iterations over which to linearly warm up the learning rate."},
        )
        lr_warmup_samples: int = field(

            metadata={"help": "Number of samples over which to linearly warm up the learning rate."},
        )
        min_lr: float = field(

            metadata={
                "help": "Minimum allowable learning rate. The scheduler will clip values below this threshold."
            },
        )
        min_wd: Optional[float] = field(

            metadata={
                "help": "Minimum value for weight decay. Defaults to the same value as `min_lr` if not specified."
            },
        )
        max_wd: Optional[float] = field(

            metadata={
                "help": "Maximum value for weight decay, scaled by the weight decay multiplier (`wd-mult`)."
            },
        )
        override_lr_scheduler: bool = field(

            metadata={
                "help": (
                    "Override the learning rate scheduler settings from a checkpoint with the values provided in the input arguments. "
                    "This includes the learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style."
                )
            },
        )
        override_max_pos_embeddings: bool = field(

            metadata={
                "help": "Use a different maximum position embedding size than specified in the checkpoint."
            },
        )
        use_checkpoint_lr_scheduler: bool = field(

            metadata={
                "help": (
                    "Use the learning rate scheduler settings saved in a checkpoint, including the learning rate, warmup iterations, "
                    "minimum learning rate, maximum number of iterations, and decay style, ignoring the input arguments."
                )
            },
        )
        override_unset_args_from_checkpoint: bool = field(

            metadata={
                "help": (
                    "Override argument values not directly supplied on the command line with values from the checkpoint, if available. "
                    "Warning: This might lead to unintended consequences."
                )
            },
        )

        def __post_init__(self) -> None:
            LR_DECAY_STYLE = ["constant", "linear", "cosine", "simple_decay"]
            if self.lr_decay_style not in LR_DECAY_STYLE:
                raise ValueError(
                    f"`lr_decay_style` should be one of {LR_DECAY_STYLE}, got {self.lr_decay_style}"
                )


    @dataclass
    class Regularization:
        attention_dropout: float = field(

            metadata={
                "help": "Dropout probability applied after the attention mechanism to prevent overfitting."
            },
        )
        hidden_dropout: float = field(

            metadata={
                "help": "Dropout probability applied to the hidden states within the transformer layers."
            },
        )
        weight_decay: float = field(

            metadata={
                "help": "Weight decay (L2 penalty) coefficient used by the AdamW optimizer to regularize the model weights."
            },
        )
        weight_decay_multiplier: float = field(

            metadata={"help": "Multiplier for weight decay applied specifically to attention layers."},
        )
        weight_decay_multiplier_embedding: float = field(

            metadata={"help": "Multiplier for weight decay applied specifically to embedding layers."},
        )
        wd_mult: float = field(

            metadata={
                "help": "Relative multiplier for weight decay, used in conjunction with the learning rate for the AdamW optimizer."
            },
        )
        wd_exp: float = field(

            metadata={
                "help": "Exponent used in the cosine schedule for weight decay with the AdamW optimizer."
            },
        )
        lr_exp: float = field(

            metadata={"help": "Exponent used in the cosine schedule for the learning rate."},
        )
        clip_grad: float = field(

            metadata={
                "help": "Maximum allowed value for the global L2 norm of gradients to prevent gradient explosion."
            },
        )
        adam_beta1: float = field(

            metadata={
                "help": "Coefficient used for computing running averages of gradient in the Adam optimizer."
            },
        )
        adam_beta2: float = field(

            metadata={
                "help": "Coefficient used for computing running averages of the squared gradient in the Adam optimizer."
            },
        )
        adam_eps: float = field(

            metadata={
                "help": "Small value added to the denominator in the Adam optimizer to improve numerical stability."
            },
        )
        sgd_momentum: float = field(

            metadata={
                "help": "Momentum factor used in the stochastic gradient descent (SGD) optimizer to accelerate gradient vectors in the right directions."
            },
        )


    @dataclass
    class Training:
        micro_batch_size: Optional[int] = field(

            metadata={
                "help": (
                    "Batch size per model instance (local batch size). "
                    "The global batch size is calculated as local batch size times data parallel size times the number of microbatches."
                )
            },
        )
        num_microbatches: Optional[int] = field(

            metadata={
                "help": (
                    "Number of microbatches. The global batch size divided by the number of microbatches equals the micro-batch size times data parallel size. "
                    "This value should be None when micro_batch_size is set."
                )
            },
        )
        micro_batch_averaging: str = field(

            metadata={
                "help": (
                    "Method for averaging loss over samples and microbatches. "
                    "'legacy' averages globally over microbatches, which can lead to a non-invariant loss under different micro-batch sizes. "
                    "'samplewise' first averages the loss over each sample, then over the batch dimension. "
                    "For full loss masks, both methods should yield the same result."
                )
            },
        )
        global_batch_size: Optional[int] = field(

            metadata={
                "help": (
                    "Total training batch size. If set, it should be a multiple of micro-batch size times data parallel size. "
                    "If not set, the global batch size defaults to micro-batch size times data parallel size, resulting in one micro-batch."
                )
            },
        )
        rampup_batch_size: Optional[List[int]] = field(

            metadata={
                "help": (
                    "Ramp up batch size with specified parameters: <start batch size> <batch size increment> <ramp-up samples>. "
                    "For example: --rampup-batch-size 16 8 300000 --global-batch-size 1024 will start with a global batch size of 16 and increase it to 1024 over 126 intervals, "
                    "using approximately 2380 samples per interval."
                )
            },
        )
        batch_size_scaling_fracs: Optional[List[float]] = field(

            metadata={
                "help": (
                    "Set points within training where batch sizes are doubled. "
                    "For example: --batch-size-scaling-fracs 0.125 0.25 0.5 --global-batch-size 256 will result in a global batch size of 256 for the last half of training samples, "
                    "128 for the quarter of samples before, 64 for one eighth of samples before that, and 32 for the first eighth of training samples."
                )
            },
        )
        recompute_granularity: Optional[str] = field(

            metadata={
                "help": (
                    "Checkpoint activations for training larger models or with larger sequences and batch sizes. "
                    "Supported granularities are: 'full' (whole transformer layer is recomputed), 'selective' (core attention part of the transformer layer is recomputed)."
                )
            },
        )
        distribute_saved_activations: bool = field(

            metadata={
                "help": "Distribute recomputed activations across model parallel groups if set to True."
            },
        )
        recompute_method: Optional[str] = field(

            metadata={
                "help": (
                    "Method for activation recomputation: "
                    "'uniform' (uniformly divide transformer layers and recompute input activations at specified granularity), "
                    "'block' (recompute input activations of only a set number of individual transformer layers per pipeline stage). "
                    "Default is to not apply recompute to any layers."
                )
            },
        )
        recompute_num_layers: int = field(

            metadata={
                "help": (
                    "Number of transformer layers to recompute: "
                    "'uniform' (number of layers in each uniformly divided recompute unit), "
                    "'block' (number of individual layers to recompute within each pipeline stage)."
                )
            },
        )
        train_tokens: Optional[int] = field(

            metadata={
                "help": "Total number of tokens for training across all training runs. Either train_iters or train_samples should be provided."
            },
        )
        train_iters: Optional[int] = field(

            metadata={
                "help": "Total number of iterations for training across all training runs. Either train_iters or train_samples should be provided."
            },
        )
        train_samples: Optional[int] = field(

            metadata={
                "help": "Total number of samples for training across all training runs. Either train_iters or train_samples should be provided."
            },
        )
        log_interval: int = field(

            metadata={
                "help": "Interval (in iterations) at which to report loss and timing information during training."
            },
        )
        exit_interval: Optional[int] = field(

            metadata={
                "help": "Exit the program after a specified number of iterations if the iteration number is divisible by this value."
            },
        )
        exit_on_metric_cond: Optional[str] = field(

            metadata={
                "help": (
                    "Exit the program if a specified condition on a metric is met. "
                    "For example, --exit-on-metric-cond='my_metric < 42' where 'my_metric' is a key in the loss dictionary."
                )
            },
        )
        exit_duration_in_mins: Optional[int] = field(
            metadata={"help": "Exit the program after a specified duration in minutes."}
        )
        exit_signal_handler: bool = field(

            metadata={
                "help": "Dynamically save a checkpoint and shut down training if a SIGTERM signal is received."
            },
        )
        masked_softmax_fusion: bool = field(

            metadata={
                "help": "Enable fusion of query-key-value scaling, masking, and softmax operations for efficiency."
            },
        )
        bias_gelu_fusion: bool = field(

            metadata={"help": "Enable fusion of bias and GELU activation functions for efficiency."},
        )
        attn_pattern: str = field(

            metadata={
                "help": (
                    "Pattern for applying attention. "
                    "Format is 'FXMY' where the first 1/X layers have attention, then every Yth layer. "
                    "Set to 'all' to enable attention in all layers, or 'none' to disable attention in all layers."
                )
            },
        )
        use_geglu: bool = field(

            metadata={"help": "Use the GeGLU activation function instead of the standard GELU."},
        )
        jit_geglu: bool = field(

            metadata={
                "help": "Enable Just-In-Time (JIT) compilation for the GeGLU activation function."
            },
        )
        ffn_hidden_size_multiplier: int = field(

            metadata={"help": "Multiplier for the hidden size in the Feed-Forward Network (FFN)."},
        )
        bias_dropout_fusion: bool = field(

            metadata={"help": "Enable fusion of bias and dropout operations for efficiency."},
        )
        optimizer: str = field( metadata={"help": "Optimizer to use for training."})
        distributed_adam_overlap_param_sync: bool = field(

            metadata={
                "help": "Overlap parameter synchronization with the forward pass for the distributed Adam optimizer."
            },
        )
        distributed_adam_overlap_grad_sync: bool = field(

            metadata={
                "help": "Overlap gradient synchronization with the backward pass for the distributed Adam optimizer."
            },
        )
        async_tensor_model_parallel_allreduce: bool = field(

            metadata={
                "help": "Enable asynchronous execution of tensor-model-parallel all-reduce with weight gradient computation of a column-linear layer."
            },
        )
        persist_layer_norm: bool = field(

            metadata={
                "help": (
                    "Enable using a persistent fused layer normalization kernel. "
                    "This kernel supports only specific hidden sizes. Check 'persist_ln_hidden_sizes' to see if your hidden size is supported."
                )
            },
        )
        sequence_parallel: bool = field(

            metadata={"help": "Enable sequence parallel optimization for efficient training."},
        )
        gradient_accumulation_fusion: bool = field(

            metadata={
                "help": "Fuse gradient accumulation with weight gradient computation in linear layers for efficiency."
            },
        )
        mode: str = field( metadata={"help": "Operation mode."})

        def __post_init__(self) -> None:
            MICRO_BATCH_AVERAGING = ["samplewise", "legacy"]
            RECOMPUTE_GRANULARITY = ["full", "selective", ""]
            RECOMPUTE_METHOD = ["uniform", "block", ""]
            OPTIMIZER = ["adam", "sgd", "distributed_adam"]
            MODE = ["train_and_eval", "eval", "train"]

            if self.micro_batch_averaging not in MICRO_BATCH_AVERAGING:
                raise ValueError(
                    f"`micro_batch_averaging` should be one of {MICRO_BATCH_AVERAGING}, got {self.micro_batch_averaging}"
                )
            if (
                self.recompute_granularity is not None
                and self.recompute_granularity not in RECOMPUTE_GRANULARITY
            ):
                raise ValueError(
                    f"`recompute_granularity` should be one of {RECOMPUTE_GRANULARITY}, got {self.recompute_granularity}"
                )
            if self.recompute_method is not None and self.recompute_method not in RECOMPUTE_METHOD:
                raise ValueError(
                    f"`recompute_method` should be one of {RECOMPUTE_METHOD}, got {self.recompute_method}"
                )
            if self.optimizer not in OPTIMIZER:
                raise ValueError(f"`optimizer` should be one of {OPTIMIZER}, got {self.optimizer}")
            if self.mode not in MODE:
                raise ValueError(f"`mode` should be one of {MODE}, got {self.mode}")


    @dataclass
    class Checkpointing:
        save: Optional[str] = field( metadata={"help": "Directory to save checkpoints."})
        experiment_dir: Optional[str] = field(
            metadata={"help": "Directory to store additional experiment information."}
        )
        save_interval: Optional[int] = field(
            metadata={"help": "Number of iterations between saving checkpoints."}
        )
        save_interval_percent: Optional[int] = field(

            metadata={
                "help": "Interval between checkpoint saves as a percentage of total consumed tokens."
            },
        )
        extra_save_interval_checkpoints_to_keep: int = field(

            metadata={"help": "Number of additional interval checkpoints to keep before deletion."},
        )
        checkpoints_at_percent: int = field(

            metadata={
                "help": (
                    "Percentage of training tokens at which to keep checkpoints. "
                    "If set to k, keeps 100/k checkpoints between 0 and the total number of training tokens. "
                    "Set to zero to disable checkpoint deletion."
                )
            },
        )
        save_optim: bool = field(
            metadata={"help": "Save the current state of the optimizer."}
        )
        save_optim_intermediate: bool = field(
            metadata={"help": "Save optimizer state only in final checkpoints."}
        )
        save_rng: bool = field(

            metadata={"help": "Save the current state of the random number generator (RNG)."},
        )
        upload: Optional[str] = field(

            metadata={
                "help": (
                    "Blob storage path to upload checkpoints. "
                    "Must be different from --download if both are specified."
                )
            },
        )
        download: Optional[str] = field(

            metadata={
                "help": (
                    "Blob storage path to download checkpoints (e.g., for starting from a pre-trained model). "
                    "Must be different from --upload if both are specified."
                )
            },
        )
        cp_cmd: str = field(

            metadata={"help": "Command to use for copying files during download/upload operations."},
        )
        upload_script: str = field(

            metadata={"help": "Script to call for uploading checkpoints to remote storage."},
        )
        load: Optional[str] = field(
            metadata={"help": "Directory containing a model checkpoint to load."}
        )
        checkpoint_iter: int = field(

            metadata={
                "help": "Specific checkpoint iteration to load. Defaults to the latest checkpoint if set to -1."
            },
        )
        dp_ckpt_save_strategy: str = field(

            metadata={"help": "Strategy for saving checkpoints across data-parallel groups."},
        )
        load_local_checkpoint: bool = field(

            metadata={
                "help": "Load a checkpoint from the local directory without downloading. Useful for debugging."
            },
        )
        load_optim: bool = field(
            metadata={"help": "Load the optimizer state when loading a checkpoint."}
        )
        load_rng: bool = field(
            metadata={"help": "Load the RNG state when loading a checkpoint."}
        )
        finetune: bool = field(

            metadata={
                "help": (
                    "Load the model for fine-tuning. Do not load optimizer or RNG state from the checkpoint and set iteration to 0. "
                    "Assumed when loading a release checkpoint."
                )
            },
        )
        prevent_random_weights: bool = field(
            metadata={"help": "Prevent starting training with random weights."}
        )
        dummy_step_after_restart: bool = field(

            metadata={"help": "Perform a single dummy forward and backward pass after restarts."},
        )
        dummy_steps_with_checksum: int = field(

            metadata={
                "help": "Perform dummy steps every N steps and compare results across data-parallel ranks for consistency."
            },
        )
        nccl_replicate_checkpoint: bool = field(

            metadata={"help": "Use NCCL to broadcast checkpoint files across the data-parallel group."},
        )
        delete_local_checkpoint_after_load: bool = field(

            metadata={
                "help": "Delete the local checkpoint file after loading if restarting after a preemption."
            },
        )
        use_tensorizer: bool = field(

            metadata={
                "help": "Use tensorizer for saving and loading checkpoints. Tensorizer helps in reducing the checkpoint size and speeding up the loading process."
            },
        )

        def __post_init__(self) -> None:
            CP_CMD = ["azcopy", "cp"]
            DP_CKPT_SAVE_STRATEGY = ["rank0", "round-robin"]

            if self.cp_cmd not in CP_CMD:
                raise ValueError(f"`cp_cmd` should be one of {CP_CMD}, got {self.cp_cmd}")
            if self.dp_ckpt_save_strategy not in DP_CKPT_SAVE_STRATEGY:
                raise ValueError(
                    f"`dp_ckpt_save_strategy` should be one of {DP_CKPT_SAVE_STRATEGY}, got {self.dp_ckpt_save_strategy}"
                )


    @dataclass
    class Autoresume:
        adlr_autoresume: bool = field(
            metadata={"help": "Enable autoresume functionality on the ADLR cluster."}
        )
        adlr_autoresume_interval: int = field(

            metadata={
                "help": "Interval (in iterations) at which to check for an autoresume termination signal."
            },
        )


    @dataclass
    class Validation:
        eval_interval: int = field(

            metadata={"help": "Number of iterations between evaluations on the validation set."},
        )


    @dataclass
    class MoE:
        moe_num_experts: Optional[int] = field(

            metadata={
                "help": "Number of experts in the Mixture of Experts (MoE) layer. Set to None to disable MoE."
            },
        )
        moe_every_n_layers: int = field(
            metadata={"help": "Apply MoE layers every n transformer layers."}
        )
        moe_top_k: int = field(
            metadata={"help": "Number of experts to select for each forward pass."}
        )
        moe_capacity_factor: float = field(

            metadata={
                "help": "Capacity factor for the MoE experts, determining the buffer size for each expert."
            },
        )
        moe_reserved_expert: bool = field(

            metadata={"help": "Reserve one expert specifically for handling overflow tokens."},
        )
        moe_overflow_algo: str = field(

            metadata={
                "help": "Algorithm used to decide which tokens to skip during overflow ('causal' or other supported methods)."
            },
        )
        moe_loss_coeff: float = field(

            metadata={
                "help": "Coefficient for the auxiliary MoE load balancing loss to ensure even distribution of tokens across experts."
            },
        )

        def __post_init__(self) -> None:
            MOE_OVERFLOW_ALGO = ["uniform", "priority", "causal"]

            if self.moe_overflow_algo not in MOE_OVERFLOW_ALGO:
                raise ValueError(
                    f"`moe_overflow_algo` should be one of {MOE_OVERFLOW_ALGO}, got {self.moe_overflow_algo}"
                )


    @dataclass
    class Biencoder:
        ict_head_size: Optional[int] = field(

            metadata={
                "help": "Dimension of block embeddings used in Inverse Cloze Task (ICT) and REALM. Default is 128 as per the paper."
            },
        )
        biencoder_projection_dim: int = field(

            metadata={
                "help": "Dimension of the projection head used in the biencoder. Default is 128 as per the paper."
            },
        )
        biencoder_shared_query_context_model: bool = field(

            metadata={"help": "Whether to share parameters between the query and context models."},
        )
        ict_load: Optional[str] = field(
            metadata={"help": "Directory containing a checkpoint for an ICTBertModel."}
        )
        bert_load: Optional[str] = field(

            metadata={
                "help": "Directory containing a checkpoint for a BertModel, required to initialize ICT and REALM."
            },
        )
        titles_data_path: Optional[str] = field(
            metadata={"help": "File path to the titles dataset used for ICT."}
        )
        query_in_block_prob: float = field(

            metadata={
                "help": "Probability of including the query within the block for the ICT dataset."
            },
        )
        use_one_sent_docs: bool = field(

            metadata={
                "help": "Whether to use single-sentence documents in ICT, instead of multi-sentence documents."
            },
        )
        evidence_data_path: Optional[str] = field(

            metadata={"help": "File path to the Wikipedia Evidence dataset from the DPR paper."},
        )
        retriever_report_topk_accuracies: List[int] = field(

            metadata={"help": "List of top-k accuracies to report (e.g., [1, 5, 20])."},
        )
        retriever_score_scaling: bool = field(

            metadata={
                "help": "Whether to scale retriever scores by the inverse square root of the hidden size for normalization."
            },
        )
        block_data_path: Optional[str] = field(

            metadata={"help": "File path to save or load BlockData for the retrieval process."},
        )
        embedding_path: Optional[str] = field(
            metadata={"help": "File path to save or load Open-Retrieval Embedding data."}
        )
        indexer_batch_size: int = field(
            metadata={"help": "Batch size for indexing jobs during the retrieval process."}
        )
        indexer_log_interval: int = field(

            metadata={"help": "Number of batches between logging progress during indexing jobs."},
        )


    @dataclass
    class ViT:
        num_classes: int = field(

            metadata={"help": "Number of output classes for the vision classification task."},
        )
        img_h: int = field(

            metadata={
                "help": "Height of the input images for the vision classification task (in pixels)."
            },
        )
        img_w: int = field(

            metadata={
                "help": "Width of the input images for the vision classification task (in pixels)."
            },
        )
        num_channels: int = field(

            metadata={
                "help": "Number of color channels in the input image data (e.g., 3 for RGB images)."
            },
        )
        patch_dim: int = field(

            metadata={
                "help": "Dimension of the patches into which the input images are divided in the Vision Transformer (ViT)."
            },
        )
        classes_fraction: float = field(

            metadata={
                "help": "Fraction of the total classes to use during training. A value less than 1.0 can be used for class subsampling."
            },
        )
        data_per_class_fraction: float = field(

            metadata={
                "help": "Fraction of the data per class to use during training. A value less than 1.0 can be used for data subsampling within each class."
            },
        )
        data_sharding: bool = field(

            metadata={
                "help": "Enable data sharding to distribute the dataset across multiple data parallel workers."
            },
        )


    @dataclass
    class Inference:
        inference_batch_times_seqlen_threshold: int = field(

            metadata={
                "help": (
                    "Threshold for deciding whether to use pipelining during inference. "
                    "If the product of batch size and sequence length is smaller than this threshold, pipelining will not be used. "
                    "If the product is greater than or equal to this threshold, pipelining will be used to improve efficiency."
                )
            },
        )



    # Hydra configuration to be overriden
    hydra: Any

    env: Environment
    logging: Logging
    data: Data
    distributed: Distributed
    mixed_precision: MixedPrecision
    transformer_engine: TransformerEngine
    network_size: NetworkSize
    initialization: Initialization
    learning_rate: LearningRate
    regularization: Regularization
    training: Training
    checkpointing: Checkpointing
    autoresume: Autoresume
    validation: Validation
    moe: MoE
    biencoder: Biencoder
    vit: ViT
    inference: Inference
