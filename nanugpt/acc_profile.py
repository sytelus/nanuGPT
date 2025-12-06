"""
Accelerator Memory Profiler - A reusable context manager for PyTorch memory profiling.

This module provides the AccProfile context manager that wraps any code block to capture
memory statistics and generate visualizations without intermediate files. It also provides
automatic annotation of model forward/backward passes and optimizer operations via hooks.

Usage:
    from acc_profile import AccProfile

    # Basic usage (profiling only)
    with AccProfile() as ap:
        # Your training/inference code here
        ...

    # With automatic annotation (recommended)
    with AccProfile(models=model, optimizers=optimizer) as ap:
        for batch in dataloader:
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(ap.memory_stats)
    ap.save_html("memory.html")

Annotation System:
    When models/optimizers are provided, AccProfile automatically registers hooks to
    annotate memory allocations with human-readable labels:

    - Model layers:     "▶FWD:layers.0.mlp.c_fc" / "◀BWD:layers.0.mlp.c_fc"
    - Optimizer ops:    "⚡OPT:AdamW.step" / "⚡OPT:AdamW.zero_grad"

    These annotations appear in memory profiler stack traces, making it easy to
    identify which operation caused each allocation.

Requirements:
    - PyTorch 2.0+ with accelerator support (CUDA, MPS, etc.)
    - pyyaml (optional, for YAML export)
"""

from __future__ import annotations

import base64
import json
import pickle
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.accelerator as accel
import torch.nn as nn
from torch.optim import Optimizer
from torch.profiler import record_function


# =============================================================================
# Annotation Format Configuration
# =============================================================================

@dataclass
class AnnotationFormat:
    """
    Configuration for how annotations appear in profiler stack traces.

    The default format produces annotations like:
        ▶FWD:model/layers.0.attention.q_proj
        ◀BWD:model/layers.0.attention.q_proj
        ⚡OPT:AdamW.step

    Attributes:
        forward_prefix: Prefix for forward pass annotations
        backward_prefix: Prefix for backward pass annotations
        optimizer_prefix: Prefix for optimizer operation annotations
        path_separator: Separator between model name and layer path
        include_class_name: If True, append (ClassName) to layer annotations
        annotate_containers: If True, annotate container modules (Sequential, ModuleList)

    Example:
        # Custom format with different symbols
        fmt = AnnotationFormat(
            forward_prefix=">>FWD:",
            backward_prefix="<<BWD:",
            optimizer_prefix="##OPT:",
        )
        with AccProfile(models=model, annotation_format=fmt) as ap:
            ...
    """
    forward_prefix: str = "▶FWD:"
    backward_prefix: str = "◀BWD:"
    optimizer_prefix: str = "⚡OPT:"
    path_separator: str = "/"
    include_class_name: bool = False
    annotate_containers: bool = False

    def format_forward(self, model_name: str, layer_path: str, class_name: str = "") -> str:
        """Format a forward pass annotation label."""
        base = f"{self.forward_prefix}{model_name}{self.path_separator}{layer_path}"
        if self.include_class_name and class_name:
            base += f"({class_name})"
        return base

    def format_backward(self, model_name: str, layer_path: str, class_name: str = "") -> str:
        """Format a backward pass annotation label."""
        base = f"{self.backward_prefix}{model_name}{self.path_separator}{layer_path}"
        if self.include_class_name and class_name:
            base += f"({class_name})"
        return base

    def format_optimizer(self, optimizer_name: str, operation: str) -> str:
        """Format an optimizer operation annotation label."""
        return f"{self.optimizer_prefix}{optimizer_name}.{operation}"


# Default annotation format instance
DEFAULT_ANNOTATION_FORMAT = AnnotationFormat()


# =============================================================================
# Hook Manager for Models and Optimizers
# =============================================================================

class _HookManager:
    """
    Internal class that manages profiling hooks for models and optimizers.

    This class handles:
    - Registering forward/backward hooks on model modules
    - Wrapping optimizer methods with profiling annotations
    - Proper cleanup of all hooks and patches on exit

    The hook manager uses record_function contexts to inject annotation labels
    into PyTorch's profiler stack traces. Each allocation that occurs within
    an annotated region will show the annotation in its stack trace.

    Implementation Notes:
        - Hooks are registered on leaf modules (no children) by default
        - Container modules can optionally be annotated via annotate_containers
        - Optimizer.step() uses built-in hooks; zero_grad() is monkey-patched
        - All contexts are tracked in _active_contexts for proper cleanup
    """

    def __init__(self, annotation_format: AnnotationFormat):
        """
        Initialize the hook manager.

        Args:
            annotation_format: Configuration for annotation label formatting
        """
        self.format = annotation_format
        self._handles: List[Any] = []  # Hook handles for removal
        self._active_contexts: Dict[Tuple[str, int], Any] = {}  # (type, id) -> context
        self._original_methods: Dict[str, Callable] = {}  # For restoring patched methods
        self._patched_foreach_ops: Dict[str, Callable] = {}  # For restoring torch._foreach_*

    def register_model(self, model: nn.Module, model_name: str = "model") -> None:
        """
        Register profiling hooks on all layers of a model.

        Registers forward pre/post hooks and backward pre/post hooks on each
        module. Annotations are injected using record_function contexts that
        span the duration of each forward/backward call.

        Args:
            model: The PyTorch model to instrument
            model_name: Name prefix for this model in annotations
        """
        for name, module in model.named_modules():
            # Determine if we should instrument this module
            has_children = len(list(module.children())) > 0
            is_root = (name == "")

            # Skip root module (will be covered by children)
            if is_root:
                continue

            # Skip containers unless explicitly enabled
            if has_children and not self.format.annotate_containers:
                continue

            layer_path = name
            class_name = module.__class__.__name__
            mod_id = id(module)

            # Create hook functions with closure over annotation labels
            self._register_module_hooks(module, model_name, layer_path, class_name, mod_id)

    def _register_module_hooks(
        self,
        module: nn.Module,
        model_name: str,
        layer_path: str,
        class_name: str,
        mod_id: int
    ) -> None:
        """Register forward and backward hooks on a single module."""

        # Forward hooks
        fwd_label = self.format.format_forward(model_name, layer_path, class_name)

        def forward_pre_hook(mod: nn.Module, args: Any) -> None:
            ctx = record_function(fwd_label)
            ctx.__enter__()
            self._active_contexts[('fwd', mod_id)] = ctx

        def forward_post_hook(mod: nn.Module, args: Any, output: Any) -> None:
            ctx = self._active_contexts.pop(('fwd', mod_id), None)
            if ctx:
                ctx.__exit__(None, None, None)

        # Backward hooks
        bwd_label = self.format.format_backward(model_name, layer_path, class_name)

        def backward_pre_hook(mod: nn.Module, grad_output: Any) -> None:
            ctx = record_function(bwd_label)
            ctx.__enter__()
            self._active_contexts[('bwd', mod_id)] = ctx

        def backward_post_hook(mod: nn.Module, grad_input: Any, grad_output: Any) -> None:
            ctx = self._active_contexts.pop(('bwd', mod_id), None)
            if ctx:
                ctx.__exit__(None, None, None)

        # Register hooks
        self._handles.append(module.register_forward_pre_hook(forward_pre_hook))
        self._handles.append(module.register_forward_hook(forward_post_hook))
        self._handles.append(module.register_full_backward_pre_hook(backward_pre_hook))
        self._handles.append(module.register_full_backward_hook(backward_post_hook))

    def register_optimizer(self, optimizer: Optimizer, optimizer_name: str = "optimizer") -> None:
        """
        Register profiling annotations on optimizer operations.

        Instruments optimizer.step() using PyTorch's built-in optimizer hooks,
        and patches optimizer.zero_grad() with a wrapper function.

        Args:
            optimizer: The PyTorch optimizer to instrument
            optimizer_name: Name for this optimizer in annotations
        """
        opt_id = id(optimizer)

        # Hook for step() using PyTorch's optimizer hooks
        step_label = self.format.format_optimizer(optimizer_name, "step")

        def step_pre_hook(opt: Optimizer, args: Any, kwargs: Any) -> None:
            ctx = record_function(step_label)
            ctx.__enter__()
            self._active_contexts[('opt_step', opt_id)] = ctx

        def step_post_hook(opt: Optimizer, args: Any, kwargs: Any) -> None:
            ctx = self._active_contexts.pop(('opt_step', opt_id), None)
            if ctx:
                ctx.__exit__(None, None, None)

        self._handles.append(optimizer.register_step_pre_hook(step_pre_hook))
        self._handles.append(optimizer.register_step_post_hook(step_post_hook))

        # Wrap zero_grad() via monkey-patching
        zero_grad_label = self.format.format_optimizer(optimizer_name, "zero_grad")
        original_zero_grad = optimizer.zero_grad
        self._original_methods[f'optimizer_{opt_id}_zero_grad'] = (optimizer, original_zero_grad)

        @wraps(original_zero_grad)
        def wrapped_zero_grad(*args: Any, **kwargs: Any) -> Any:
            with record_function(zero_grad_label):
                return original_zero_grad(*args, **kwargs)

        optimizer.zero_grad = wrapped_zero_grad  # type: ignore[method-assign]

    def cleanup(self) -> None:
        """
        Remove all registered hooks and restore patched methods.

        This must be called when profiling is complete to avoid memory leaks
        and restore original optimizer behavior.
        """
        # Remove all hook handles
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        # Close any remaining active contexts
        for ctx in self._active_contexts.values():
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._active_contexts.clear()

        # Restore patched optimizer methods
        for key, (optimizer, original_method) in self._original_methods.items():
            if 'zero_grad' in key:
                optimizer.zero_grad = original_method  # type: ignore[method-assign]
        self._original_methods.clear()

        # Restore patched foreach operations
        for op_name, original_fn in self._patched_foreach_ops.items():
            setattr(torch, op_name, original_fn)
        self._patched_foreach_ops.clear()


# =============================================================================
# HTML Visualization Template with Enhanced Styling
# =============================================================================

_MEMORY_VIZ_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Profile</title>
    <style>
        /* Enhanced styling for annotation visibility */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 0;
        }

        /* Annotation legend */
        .annotation-legend {
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            max-width: 300px;
        }

        .annotation-legend h4 {
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 6px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 6px 0;
        }

        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
            flex-shrink: 0;
        }

        .legend-forward { background: linear-gradient(135deg, #4CAF50, #2E7D32); }
        .legend-backward { background: linear-gradient(135deg, #2196F3, #1565C0); }
        .legend-optimizer { background: linear-gradient(135deg, #FF9800, #E65100); }

        .legend-text {
            color: #555;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }

        /* Stack trace annotation highlighting */
        .annotation-highlight-forward {
            background: linear-gradient(90deg, rgba(76, 175, 80, 0.2), transparent);
            border-left: 3px solid #4CAF50;
            padding-left: 8px;
            margin: 2px 0;
            font-weight: 600;
        }

        .annotation-highlight-backward {
            background: linear-gradient(90deg, rgba(33, 150, 243, 0.2), transparent);
            border-left: 3px solid #2196F3;
            padding-left: 8px;
            margin: 2px 0;
            font-weight: 600;
        }

        .annotation-highlight-optimizer {
            background: linear-gradient(90deg, rgba(255, 152, 0, 0.2), transparent);
            border-left: 3px solid #FF9800;
            padding-left: 8px;
            margin: 2px 0;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <!-- Annotation Legend -->
    <div class="annotation-legend">
        <h4>📊 Annotation Legend</h4>
        <div class="legend-item">
            <div class="legend-color legend-forward"></div>
            <span class="legend-text">▶FWD: Forward Pass</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-backward"></div>
            <span class="legend-text">◀BWD: Backward Pass</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-optimizer"></div>
            <span class="legend-text">⚡OPT: Optimizer</span>
        </div>
    </div>

    <!-- Memory Visualization -->
    <script type="module">
        import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
        const local_files = $SNAPSHOT
        add_local_files(local_files, $VIZ_KIND)

        // Post-process to highlight annotations in stack traces
        setTimeout(() => {
            const highlightAnnotations = () => {
                // Find all text elements that might contain stack traces
                const allText = document.body.getElementsByTagName('*');
                for (let elem of allText) {
                    if (elem.children.length === 0 && elem.textContent) {
                        const text = elem.textContent;
                        // Check for annotation markers
                        if (text.includes('▶FWD:') || text.includes('>>FWD:')) {
                            elem.classList.add('annotation-highlight-forward');
                        } else if (text.includes('◀BWD:') || text.includes('<<BWD:')) {
                            elem.classList.add('annotation-highlight-backward');
                        } else if (text.includes('⚡OPT:') || text.includes('##OPT:')) {
                            elem.classList.add('annotation-highlight-optimizer');
                        }
                    }
                }
            };

            // Run highlighting periodically to catch dynamically added content
            highlightAnnotations();
            const observer = new MutationObserver(highlightAnnotations);
            observer.observe(document.body, { childList: true, subtree: true });
        }, 1000);
    </script>
</body>
</html>
"""


def _format_viz(data: Dict[str, Any], viz_kind: str = "Active Memory Timeline") -> str:
    """
    Convert snapshot data to an HTML visualization string.

    This embeds the pickled snapshot as base64 in the HTML, which is then
    rendered client-side using PyTorch's MemoryViz.js library. The HTML
    includes enhanced styling for annotation visibility.

    Args:
        data: Memory snapshot dictionary from torch.cuda.memory._snapshot()
        viz_kind: Type of visualization (e.g., "Active Memory Timeline")

    Returns:
        Complete HTML document as a string
    """
    buffer = pickle.dumps(data)
    # Pad to multiple of 3 for base64 encoding
    buffer += b"\x00" * (3 - len(buffer) % 3)
    encoded = base64.b64encode(buffer).decode("utf-8")
    payload = json.dumps([{"name": "snapshot.pickle", "base64": encoded}])
    return _MEMORY_VIZ_TEMPLATE.replace("$VIZ_KIND", repr(viz_kind)).replace("$SNAPSHOT", payload)


# =============================================================================
# Main AccProfile Context Manager
# =============================================================================

# Type alias for model/optimizer inputs (single object or sequence)
ModelInput = Union[nn.Module, Sequence[nn.Module], None]
OptimizerInput = Union[Optimizer, Sequence[Optimizer], None]


class AccProfile:
    """
    Context manager for profiling accelerator memory usage with automatic annotations.

    AccProfile captures memory statistics and allocation history, optionally
    instrumenting models and optimizers with profiling hooks for enhanced
    stack trace readability.

    Features:
        - Memory statistics collection (peak, allocated, reserved, etc.)
        - Allocation history recording for visualization
        - Automatic forward/backward pass annotations on models
        - Automatic optimizer operation annotations
        - Configurable annotation format
        - Enhanced HTML visualization with annotation highlighting

    Attributes:
        memory_stats: Dictionary of memory statistics after profiling
        profile_data: Raw snapshot data suitable for visualization/serialization
        device: The accelerator device being profiled

    Example:
        # Basic profiling
        with AccProfile() as ap:
            output = model(input)
        print(ap.memory_stats)

        # With automatic annotations (recommended for training)
        model = MyModel().cuda()
        optimizer = torch.optim.AdamW(model.parameters())

        with AccProfile(models=model, optimizers=optimizer) as ap:
            for batch in dataloader:
                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ap.save_html("training_profile.html")

        # Multiple models and optimizers
        with AccProfile(
            models=[encoder, decoder],
            optimizers=[enc_opt, dec_opt],
            model_names=["encoder", "decoder"],
            optimizer_names=["enc_opt", "dec_opt"]
        ) as ap:
            ...

        # Custom annotation format
        custom_format = AnnotationFormat(
            forward_prefix=">>FWD:",
            backward_prefix="<<BWD:",
            include_class_name=True
        )
        with AccProfile(models=model, annotation_format=custom_format) as ap:
            ...
    """

    def __init__(
        self,
        profile_memory: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        models: ModelInput = None,
        optimizers: OptimizerInput = None,
        model_names: Optional[Union[str, Sequence[str]]] = None,
        optimizer_names: Optional[Union[str, Sequence[str]]] = None,
        annotation_format: Optional[AnnotationFormat] = None,
    ):
        """
        Initialize the profiler.

        Args:
            profile_memory: If True, record allocation history for visualization.
                           If False, only collect summary statistics.
            device: Target device. If None, uses current accelerator.
            models: Single model or list of models to instrument with hooks.
                   If None, no model hooks are registered.
            optimizers: Single optimizer or list of optimizers to instrument.
                       If None, no optimizer hooks are registered.
            model_names: Names for models in annotations. Can be a single string
                        (used for single model) or list of strings matching models.
                        If None, defaults to "model", "model_1", "model_2", etc.
            optimizer_names: Names for optimizers in annotations. Same format as
                            model_names. If None, uses optimizer class name.
            annotation_format: Custom format for annotation labels.
                              If None, uses DEFAULT_ANNOTATION_FORMAT.
        """
        self.profile_memory = profile_memory
        self.device = (
            torch.device(device) if device
            else torch.device(accel.current_accelerator() or "cpu")
        )

        # Normalize inputs to lists
        self._models = self._to_list(models)
        self._optimizers = self._to_list(optimizers)
        self._model_names = self._make_names(model_names, self._models, "model")
        self._optimizer_names = self._make_names(
            optimizer_names,
            self._optimizers,
            default_factory=lambda i, opt: opt.__class__.__name__
        )

        self._format = annotation_format or DEFAULT_ANNOTATION_FORMAT
        self._hook_manager: Optional[_HookManager] = None

        # Output attributes
        self.memory_stats: Dict[str, Any] = {}
        self.profile_data: Optional[Dict[str, Any]] = None
        self._is_cuda = self.device.type == "cuda"

    @staticmethod
    def _to_list(obj: Any) -> List[Any]:
        """Convert single object or sequence to list, handling None."""
        if obj is None:
            return []
        if isinstance(obj, (list, tuple)):
            return list(obj)
        return [obj]

    @staticmethod
    def _make_names(
        names: Optional[Union[str, Sequence[str]]],
        objects: List[Any],
        default: str = "item",
        default_factory: Optional[Callable[[int, Any], str]] = None
    ) -> List[str]:
        """Generate names list matching objects list."""
        if not objects:
            return []

        if names is None:
            # Generate default names
            if default_factory:
                return [default_factory(i, obj) for i, obj in enumerate(objects)]
            if len(objects) == 1:
                return [default]
            return [f"{default}_{i}" for i in range(len(objects))]

        if isinstance(names, str):
            names = [names]

        names = list(names)

        # Pad with numbered defaults if needed
        while len(names) < len(objects):
            idx = len(names)
            if default_factory:
                names.append(default_factory(idx, objects[idx]))
            else:
                names.append(f"{default}_{idx}")

        return names[:len(objects)]

    def __enter__(self) -> AccProfile:
        """Start profiling: reset stats, register hooks, begin recording."""
        accel.memory.empty_cache()
        accel.memory.reset_peak_memory_stats()
        accel.memory.reset_accumulated_memory_stats()

        # Register annotation hooks if models or optimizers provided
        if self._models or self._optimizers:
            self._hook_manager = _HookManager(self._format)

            for model, name in zip(self._models, self._model_names):
                self._hook_manager.register_model(model, name)

            for optimizer, name in zip(self._optimizers, self._optimizer_names):
                self._hook_manager.register_optimizer(optimizer, name)

        if self.profile_memory and self._is_cuda:
            # Start recording allocation history with stack traces
            torch.cuda.memory._record_memory_history(max_entries=100000)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop profiling: synchronize, collect stats, capture snapshot, cleanup hooks."""
        accel.synchronize()
        self.memory_stats = dict(accel.memory.memory_stats())

        if self.profile_memory and self._is_cuda:
            # Capture the memory snapshot before stopping recording
            self.profile_data = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)

        # Cleanup hooks
        if self._hook_manager:
            self._hook_manager.cleanup()
            self._hook_manager = None

        accel.memory.empty_cache()

    def to_html(self, viz_kind: str = "Active Memory Timeline") -> str:
        """
        Generate HTML visualization from captured profile data.

        The generated HTML includes:
        - Interactive memory timeline visualization
        - Annotation legend explaining Forward/Backward/Optimizer markers
        - Automatic highlighting of annotations in stack traces

        Args:
            viz_kind: Visualization type. Options:
                - "Active Memory Timeline" (default): Shows allocations over time
                - "Allocator State History": Shows segment packing at each point

        Returns:
            HTML document string

        Raises:
            ValueError: If no profile data was captured
        """
        if self.profile_data is None:
            raise ValueError(
                "No profile data available. Ensure profile_memory=True and device is CUDA."
            )
        return _format_viz(self.profile_data, viz_kind)

    def save_html(self, path: str, viz_kind: str = "Active Memory Timeline") -> str:
        """
        Save memory visualization as an HTML file.

        Args:
            path: Output file path (e.g., "memory_profile.html")
            viz_kind: Visualization type (see to_html for options)

        Returns:
            The path where the file was saved
        """
        with open(path, "w") as f:
            f.write(self.to_html(viz_kind))
        return path

    def save_yaml(self, path: str) -> str:
        """
        Save memory statistics and profile metadata as YAML.

        Exports memory_stats and a summary of profile_data (without full binary data)
        in a human-readable format.

        Args:
            path: Output file path (e.g., "memory_stats.yaml")

        Returns:
            The path where the file was saved
        """
        import yaml

        output = {
            "memory_stats": self.memory_stats,
            "profile_summary": self._get_profile_summary() if self.profile_data else None,
            "annotation_config": {
                "forward_prefix": self._format.forward_prefix,
                "backward_prefix": self._format.backward_prefix,
                "optimizer_prefix": self._format.optimizer_prefix,
            }
        }
        with open(path, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        return path

    def save_pickle(self, path: str) -> str:
        """
        Save raw profile data as a pickle file.

        This format is compatible with PyTorch's memory visualization tools:
            python -m torch.utils.viz._memory_viz trace_plot memory.pkl -o memory.html

        Args:
            path: Output file path (e.g., "memory_snapshot.pickle")

        Returns:
            The path where the file was saved

        Raises:
            ValueError: If no profile data was captured
        """
        if self.profile_data is None:
            raise ValueError(
                "No profile data available. Ensure profile_memory=True and device is CUDA."
            )
        with open(path, "wb") as f:
            pickle.dump(self.profile_data, f)
        return path

    def _get_profile_summary(self) -> Dict[str, Any]:
        """Extract a lightweight summary from profile_data for YAML export."""
        if not self.profile_data:
            return {}

        segments = self.profile_data.get("segments", [])
        traces = self.profile_data.get("device_traces", [])

        return {
            "num_segments": len(segments),
            "total_segment_size": sum(s.get("total_size", 0) for s in segments),
            "num_devices_with_traces": sum(1 for t in traces if t),
            "total_trace_events": sum(len(t) for t in traces),
        }

    @property
    def peak_memory_gb(self) -> float:
        """Get peak allocated memory in gigabytes."""
        return self.memory_stats.get("allocated_bytes.all.peak", 0) / 1e9

    @property
    def current_memory_gb(self) -> float:
        """Get current allocated memory in gigabytes."""
        return self.memory_stats.get("allocated_bytes.all.current", 0) / 1e9

    def __repr__(self) -> str:
        models_info = f", models={len(self._models)}" if self._models else ""
        opts_info = f", optimizers={len(self._optimizers)}" if self._optimizers else ""
        return (
            f"AccProfile(device={self.device}, "
            f"peak_allocated={self.peak_memory_gb:.3f} GB{models_info}{opts_info})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_simple_format() -> AnnotationFormat:
    """Create a simple ASCII-only annotation format."""
    return AnnotationFormat(
        forward_prefix=">>FWD:",
        backward_prefix="<<BWD:",
        optimizer_prefix="##OPT:",
    )


def create_verbose_format() -> AnnotationFormat:
    """Create a verbose annotation format with class names."""
    return AnnotationFormat(
        forward_prefix="▶FWD:",
        backward_prefix="◀BWD:",
        optimizer_prefix="⚡OPT:",
        include_class_name=True,
        annotate_containers=True,
    )