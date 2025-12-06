"""
Accelerator Memory Profiler - A reusable context manager for PyTorch memory profiling.

This module provides the AccMemProfiler context manager that wraps any code block to capture
memory statistics and generate visualizations without intermediate files.

Usage:
    from acc_profile import AccMemProfiler

    with AccMemProfiler(profile_memory=True) as ap:
        # Your training/inference code here
        ...

    print(ap.memory_stats)    # Memory statistics dictionary
    print(ap.profile_data)    # Raw snapshot data (can be pickled)
    ap.save_html("memory.html")  # Interactive visualization
    ap.save_yaml("memory.yaml")  # Export stats as YAML
    ap.save_pickle("memory.pkl") # Export snapshot as pickle

Requirements:
    - PyTorch with accelerator support (CUDA, MPS, etc.)
    - pyyaml (for YAML export)
"""

from __future__ import annotations

import base64
import json
import pickle
from typing import Any

import torch
import torch.accelerator as accel


# HTML template for memory visualization (from PyTorch's _memory_viz.py)
_MEMORY_VIZ_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head><title>Memory Profile</title></head>
<body>
<script type="module">
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
const local_files = $SNAPSHOT
add_local_files(local_files, $VIZ_KIND)
</script>
</body>
</html>
"""


def _format_viz(data: dict[str, Any], viz_kind: str = "Active Memory Timeline") -> str:
    """
    Convert snapshot data to an HTML visualization string.

    This embeds the pickled snapshot as base64 in the HTML, which is then
    rendered client-side using PyTorch's MemoryViz.js library.

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


class AccMemProfiler:
    """
    Context manager for profiling accelerator memory usage.

    Captures memory statistics and optionally records allocation history
    for visualization. Works with any PyTorch-supported accelerator (CUDA, MPS, etc.).

    Attributes:
        memory_stats: Dictionary of memory statistics after profiling
        profile_data: Raw snapshot data suitable for visualization/serialization
        device: The accelerator device being profiled

    Example:
        with AccMemProfiler(profile_memory=True) as ap:
            model = MyModel().to("cuda")
            output = model(input_data)

        print(f"Peak memory: {ap.memory_stats.get('allocated_bytes.all.peak', 0) / 1e9:.2f} GB")
        ap.save_html("profile.html")
    """

    def __init__(self, profile_memory: bool = True, device: torch.device | str | None = None):
        """
        Initialize the profiler.

        Args:
            profile_memory: If True, record allocation history for visualization.
                           If False, only collect summary statistics.
            device: Target device. If None, uses current accelerator.
        """
        self.profile_memory = profile_memory
        self.device = torch.device(device) if device else torch.device(
            accel.current_accelerator() or "cpu"
        )
        self.memory_stats: dict[str, Any] = {}
        self.profile_data: dict[str, Any] | None = None
        self._is_cuda = self.device.type == "cuda"

    def __enter__(self) -> AccMemProfiler:
        """Start profiling: reset stats and optionally begin recording history."""
        accel.memory.empty_cache()
        accel.memory.reset_peak_memory_stats()
        accel.memory.reset_accumulated_memory_stats()

        if self.profile_memory and self._is_cuda:
            # Start recording allocation history with stack traces
            torch.cuda.memory._record_memory_history(max_entries=100000)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop profiling: synchronize, collect stats, and capture snapshot."""
        accel.synchronize()
        self.memory_stats = dict(accel.memory.memory_stats())

        if self.profile_memory and self._is_cuda:
            # Capture the memory snapshot before stopping recording
            self.profile_data = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)

        accel.memory.empty_cache()

    def to_html(self, viz_kind: str = "Active Memory Timeline") -> str:
        """
        Generate HTML visualization from captured profile data.

        Args:
            viz_kind: Visualization type. Options:
                - "Active Memory Timeline" (default): Shows allocations over time
                - "Allocator State History": Shows segment packing at each point

        Returns:
            HTML document string

        Raises:
            ValueError: If no profile data was captured (profile_memory was False or device is not CUDA)
        """
        if self.profile_data is None:
            raise ValueError("No profile data available. Ensure profile_memory=True and device is CUDA.")
        return _format_viz(self.profile_data, viz_kind)

    def save_html(self, path: str, viz_kind: str = "Active Memory Timeline") -> str:
        """
        Save memory visualization as an HTML file.

        Args:
            path: Output file path (e.g., "memory_profile.html")
            viz_kind: Visualization type (see to_html for options)
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
        """
        import yaml

        output = {
            "memory_stats": self.memory_stats,
            "profile_summary": self._get_profile_summary() if self.profile_data else None,
        }
        with open(path, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        return path

    def save_pickle(self, path: str) -> str:
        """
        Save raw profile data as a pickle file.

        This format is compatible with PyTorch's memory visualization tools:
            python _memory_viz.py trace_plot memory.pickle -o memory.html

        Args:
            path: Output file path (e.g., "memory_snapshot.pickle")

        Raises:
            ValueError: If no profile data was captured
        """
        if self.profile_data is None:
            raise ValueError("No profile data available. Ensure profile_memory=True and device is CUDA.")
        with open(path, "wb") as f:
            pickle.dump(self.profile_data, f)

        return path

    def _get_profile_summary(self) -> dict[str, Any]:
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

    def __repr__(self) -> str:
        peak = self.memory_stats.get("allocated_bytes.all.peak", 0)
        return f"AccMemProfiler(device={self.device}, peak_allocated={peak / 1e9:.3f} GB)"