#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
volcano_ns_node_stats.py

Produce node availability and schedulability stats for a Volcano-based GPU cluster,
without metrics-server and without cluster-wide permissions.

All queries are strictly limited to the namespace provided via VOLCANO_NAMESPACE.

How it works:
- Lists Pods and Events in the namespace.
- Parses scheduler "FailedScheduling" events to infer:
    "X/Y nodes are available"  -> X (available), Y (total considered)
    and the detailed reason buckets (e.g., "Insufficient nvidia.com/gpu").
- Builds CPU vs GPU workload views:
    * CPU: Pods requesting 0 NVIDIA GPUs
    * GPU: Pods requesting >= 1 NVIDIA GPU
- Augments with:
    * Observed nodes currently hosting this namespace's Pods, split CPU vs GPU by
      the resources those Pods request (namespace-scoped inference).
    * ResourceQuota usage for CPU/GPU if present.

Output example (structure):
Schedulability (node counts)
  KIND  SCHED          COUNT
  ----  -------------  -----
  CPU   Schedulable    <X_cpu>
  CPU   Unschedulable  <Y_cpu - X_cpu>
  GPU   Schedulable    <X_gpu>
  GPU   Unschedulable  <Y_gpu - X_gpu>

Availability among schedulable worker nodes (node counts)
  KIND  STATE                           COUNT
  ----  ------------------------------  -----
  CPU   Available                       <X_cpu>
  CPU   Full (capacity-limited)         <insufficient cpu>
  CPU   Other unschedulable reasons     <...>
  GPU   Available                       <X_gpu>
  GPU   Full (capacity-limited/absent)  <insufficient nvidia.com/gpu>
  GPU   Other unschedulable reasons     <...>

Notes:
- "Full (capacity-limited/absent)" for GPU includes nodes with no GPUs or not enough GPUs.
- Counts come from the most recent FailedScheduling events for your pending Pods.
"""

import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from kubernetes import client, config
from kubernetes.client import ApiException


# ------------------------------- Utilities -------------------------------- #

def load_kube():
    """Load kube config (in-cluster first, fallback to local kubeconfig)."""
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()


def env_namespace() -> str:
    ns = os.environ.get("VOLCANO_NAMESPACE")
    if not ns:
        print("ERROR: VOLCANO_NAMESPACE environment variable is not set.", file=sys.stderr)
        sys.exit(2)
    return ns


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def human_ts(ts: Optional[datetime]) -> str:
    if not ts:
        return "unknown"
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def pad(s: str, w: int) -> str:
    return s + (" " * max(0, w - len(s)))


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Header
    header_line = "  " + "  ".join(pad(h, widths[i]) for i, h in enumerate(headers))
    sep_line = "  " + "  ".join("-" * widths[i] for i, _ in enumerate(headers))
    print(header_line)
    print(sep_line)
    # Rows
    for row in rows:
        print("  " + "  ".join(pad(str(cell), widths[i]) for i, cell in enumerate(row)))


# ----------------------------- Kubernetes I/O ------------------------------ #

def list_ns_pods(v1: client.CoreV1Api, ns: str) -> List[client.V1Pod]:
    resp = v1.list_namespaced_pod(ns)
    return resp.items or []


def list_ns_resourcequotas(v1: client.CoreV1Api, ns: str) -> List[client.V1ResourceQuota]:
    try:
        resp = v1.list_namespaced_resource_quota(ns)
        return resp.items or []
    except ApiException as e:
        if e.status == 403:
            return []
        raise


def list_ns_podgroups(ns: str) -> List[dict]:
    """
    Try to list Volcano PodGroups (namespaced CRD).
    If CRD not installed or no access, return empty.

    GVR: scheduling.volcano.sh/v1beta1, kind: PodGroup (commonly)
    """
    try:
        api = client.CustomObjectsApi()
        # The plural may be "podgroups" for Volcano.
        resp = api.list_namespaced_custom_object(
            group="scheduling.volcano.sh", version="v1beta1",
            namespace=ns, plural="podgroups"
        )
        items = resp.get("items", [])
        return items
    except ApiException:
        return []
    except Exception:
        return []


def list_ns_events(ns: str) -> List[dict]:
    """
    Return events (as dicts) for core/v1 and events.k8s.io/v1.
    We'll normalize a minimal schema: {reason, message/note, type, involved_object_name, last_timestamp}
    """
    out: List[dict] = []

    # Try events.k8s.io/v1 first
    try:
        ev1 = client.EventsV1Api()
        resp = ev1.list_namespaced_event(ns)
        for e in resp.items or []:
            # events.k8s.io/v1 fields
            reason = getattr(e, "reason", "") or ""
            note = getattr(e, "note", "") or ""
            etype = getattr(e, "type", "") or ""
            regarding = getattr(e, "regarding", None)
            obj_name = getattr(regarding, "name", "") if regarding else ""
            # timestamps
            # Try 'event_time' first, then 'series', then 'deprecated_last_timestamp'
            ts = None
            if getattr(e, "event_time", None):
                ts = e.event_time
            elif getattr(e, "series", None) and getattr(e.series, "last_observed_time", None):
                ts = e.series.last_observed_time
            elif getattr(e, "deprecated_last_timestamp", None):
                ts = e.deprecated_last_timestamp
            out.append({
                "reason": reason,
                "message": note,
                "type": etype,
                "involved_object_name": obj_name,
                "last_timestamp": ts
            })
    except ApiException:
        pass
    except Exception:
        pass

    # Fallback: core/v1 events
    try:
        v1 = client.CoreV1Api()
        resp = v1.list_namespaced_event(ns)
        for e in resp.items or []:
            reason = getattr(e, "reason", "") or ""
            msg = getattr(e, "message", "") or ""
            etype = getattr(e, "type", "") or ""
            inv = getattr(e, "involved_object", None)
            obj_name = getattr(inv, "name", "") if inv else ""
            ts = getattr(e, "last_timestamp", None) or getattr(e, "event_time", None)
            out.append({
                "reason": reason,
                "message": msg,
                "type": etype,
                "involved_object_name": obj_name,
                "last_timestamp": ts
            })
    except ApiException:
        pass
    except Exception:
        pass

    # Sort newest first
    def ts_key(d: dict):
        ts = d.get("last_timestamp")
        if not ts:
            return 0
        try:
            return int(ts.timestamp())
        except Exception:
            # Some clients return str timestamps; attempt parse
            try:
                return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except Exception:
                return 0

    out.sort(key=ts_key, reverse=True)
    return out


# -------------------------- Parsing & Aggregation -------------------------- #

AVAIL_RE = re.compile(r'(\d+)\s*/\s*(\d+)\s+nodes?\s+(?:are|were)\s+available', re.IGNORECASE)
COUNT_PREFIX_RE = re.compile(r'^\s*(\d+)\s+(.+?)\s*$')

def sum_gpu_requests_in_pod(p: client.V1Pod) -> int:
    """Sum requested NVIDIA GPUs across all containers/initContainers (requests or limits)."""
    def get_gpu(res):
        if not res:
            return 0
        req = 0
        if res.requests and "nvidia.com/gpu" in res.requests:
            try:
                req = max(req, int(res.requests["nvidia.com/gpu"]))
            except Exception:
                pass
        if res.limits and "nvidia.com/gpu" in res.limits:
            try:
                req = max(req, int(res.limits["nvidia.com/gpu"]))
            except Exception:
                pass
        return req

    total = 0
    for c in (p.spec.containers or []):
        total += get_gpu(c.resources)
    for c in (p.spec.init_containers or []):
        total = max(total, get_gpu(c.resources))
    return total


def is_gpu_pod(p: client.V1Pod) -> bool:
    return sum_gpu_requests_in_pod(p) > 0


def is_pending(p: client.V1Pod) -> bool:
    # Pending (includes unscheduled or pulling images)
    return (p.status and p.status.phase == "Pending")


def parse_failed_scheduling_message(msg: str) -> Tuple[Optional[int], Optional[int], Dict[str, int]]:
    """
    From a message like:
      "0/8 nodes are available: 3 Insufficient cpu, 5 Insufficient memory."
    Return (available, total, reasons_dict)
    reasons_dict keys are normalized like:
      "insufficient/cpu", "insufficient/memory", "insufficient/nvidia.com/gpu",
      "taints", "node_selector_mismatch", "other/<text>"
    """
    if not msg:
        return None, None, {}

    available, total = None, None
    m = AVAIL_RE.search(msg)
    if m:
        available = safe_int(m.group(1))
        total = safe_int(m.group(2))

    reasons: Dict[str, int] = {}
    # Everything after the colon often contains the reason fragments
    parts = msg.split(":", 1)
    if len(parts) == 2:
        tail = parts[1]
        # Split by comma, then parse each "N reason text"
        frags = [t.strip(" .") for t in tail.split(",") if t.strip()]
        for frag in frags:
            m2 = COUNT_PREFIX_RE.match(frag)
            if not m2:
                continue
            count = safe_int(m2.group(1))
            text = m2.group(2).lower()

            key = None
            if text.startswith("insufficient "):
                res = text.replace("insufficient", "", 1).strip()
                key = f"insufficient/{res}"
            elif "had taint {" in text:
                key = "taints"
            elif "didn't match node selector" in text or "did not match node selector" in text:
                key = "node_selector_mismatch"
            elif "were unschedulable" in text:
                key = "unschedulable"  # commonly cordoned/drained
            else:
                # normalize some other common scheduler texts
                if "preemption is not helpful" in text:
                    key = "preemption_not_helpful"
                elif "ineligible" in text:
                    key = "ineligible"
                else:
                    key = f"other/{text[:48]}"

            reasons[key] = reasons.get(key, 0) + count

    return available, total, reasons


def pick_representative_failed_event(
    events: List[dict],
    pod_names: set
) -> Optional[dict]:
    """
    Choose the newest FailedScheduling event for any pod in pod_names.
    """
    for e in events:
        if e.get("reason") == "FailedScheduling" and e.get("involved_object_name") in pod_names:
            return e
    return None


def aggregate_schedulability_for_kind(
    pods: List[client.V1Pod],
    events: List[dict],
    kind: str  # "CPU" or "GPU"
) -> Tuple[Optional[int], Optional[int], Dict[str, int], Optional[dict]]:
    """
    Return (available, total, reasons, used_event) for the given kind.
    If no pending pod of that kind has recent FailedScheduling, returns Nones.
    """
    if kind not in ("CPU", "GPU"):
        raise ValueError("kind must be CPU or GPU")

    if kind == "GPU":
        kind_pods = [p for p in pods if is_gpu_pod(p)]
    else:
        kind_pods = [p for p in pods if not is_gpu_pod(p)]

    # Consider only Pending pods of that kind for FailedScheduling
    pending_names = {p.metadata.name for p in kind_pods if is_pending(p)}
    if not pending_names:
        # Still try: sometimes pods have transitioned, but recent FailedScheduling events remain
        # We'll use ANY pod of that kind to look up the newest FailedScheduling event.
        pending_names = {p.metadata.name for p in kind_pods}

    if not pending_names:
        return None, None, {}, None

    e = pick_representative_failed_event(events, pending_names)
    if not e:
        return None, None, {}, None

    available, total, reasons = parse_failed_scheduling_message(e.get("message", "") or e.get("note", ""))
    return available, total, reasons, e


# ------------------------------- Main Report ------------------------------- #

def main():
    ns = env_namespace()
    load_kube()

    v1 = client.CoreV1Api()
    pods = list_ns_pods(v1, ns)
    events = list_ns_events(ns)
    quotas = list_ns_resourcequotas(v1, ns)
    podgroups = list_ns_podgroups(ns)

    ts = now_utc_iso()

    # Observed nodes for this namespace
    observed_nodes: Dict[str, Dict[str, Any]] = {}
    for p in pods:
        node = (p.spec.node_name or "").strip()
        if not node:
            continue
        rec = observed_nodes.setdefault(node, {"has_gpu_pod": False, "pod_names": set()})
        rec["pod_names"].add(p.metadata.name)
        if is_gpu_pod(p):
            rec["has_gpu_pod"] = True

    # Aggregations for CPU/GPU
    cpu_avail, cpu_total, cpu_reasons, cpu_ev = aggregate_schedulability_for_kind(pods, events, "CPU")
    gpu_avail, gpu_total, gpu_reasons, gpu_ev = aggregate_schedulability_for_kind(pods, events, "GPU")

    # Pending pods quick summary
    pending_gpu = [p for p in pods if is_gpu_pod(p) and is_pending(p)]
    pending_cpu = [p for p in pods if (not is_gpu_pod(p)) and is_pending(p)]

    # --------------------------- Print the report --------------------------- #
    print(f"Namespace: {ns}")
    print(f"Generated at: {ts}")
    print()

    # Schedulability (node counts)
    print("Schedulability (node counts)")
    sched_rows = []
    if cpu_total is not None and cpu_avail is not None:
        sched_rows.append(["CPU", "Schedulable", cpu_avail])
        sched_rows.append(["CPU", "Unschedulable", cpu_total - cpu_avail])
    else:
        sched_rows.append(["CPU", "Schedulable", "N/A"])
        sched_rows.append(["CPU", "Unschedulable", "N/A"])

    # ControlPlane row is omitted by default (requires node-scope or strong heuristics).
    if gpu_total is not None and gpu_avail is not None:
        sched_rows.append(["GPU", "Schedulable", gpu_avail])
        sched_rows.append(["GPU", "Unschedulable", gpu_total - gpu_avail])
    else:
        sched_rows.append(["GPU", "Schedulable", "N/A"])
        sched_rows.append(["GPU", "Unschedulable", "N/A"])

    print_table(["KIND", "SCHED", "COUNT"], sched_rows)
    print()

    # Availability among Schedulable worker nodes
    # We interpret:
    #   Available = X in "X/Y nodes are available"
    #   Full (capacity-limited) = count of "Insufficient <resource>" for that kind
    #   Other unschedulable reasons = remainder
    print("Availability among Schedulable worker nodes (node counts)")
    avail_rows = []

    def breakdown(avail, total, reasons, kind):
        if avail is None or total is None:
            return [["%s" % kind, "Available", "N/A"],
                    ["%s" % kind, "Full (capacity-limited%s)" % ("/absent" if kind == "GPU" else ""), "N/A"],
                    ["%s" % kind, "Other unschedulable reasons", "N/A"]]
        # "Full" reasons
        if kind == "GPU":
            full = reasons.get("insufficient/nvidia.com/gpu", 0)
            full_label = "Full (capacity-limited/absent)"
        else:
            full = reasons.get("insufficient/cpu", 0)
            full_label = "Full (capacity-limited)"
        other = (total - avail) - full
        if other < 0:
            other = 0
        return [[kind, "Available", avail],
                [kind, full_label, full],
                [kind, "Other unschedulable reasons", other]]

    avail_rows += breakdown(cpu_avail, cpu_total, cpu_reasons, "CPU")
    avail_rows += breakdown(gpu_avail, gpu_total, gpu_reasons, "GPU")

    print_table(["KIND", "STATE", "COUNT"], avail_rows)
    print()

    # Observed nodes (namespace-scoped inference)
    total_observed = len(observed_nodes)
    gpu_observed = sum(1 for n, rec in observed_nodes.items() if rec["has_gpu_pod"])
    cpu_observed = total_observed - gpu_observed
    print("Observed nodes currently running this namespace's Pods")
    print_table(["METRIC", "VALUE"], [
        ["Distinct nodes observed", total_observed],
        ["Nodes observed with GPU Pods (ns)", gpu_observed],
        ["Nodes observed with only CPU Pods (ns)", cpu_observed],
    ])
    print()

    # Pending Pods summary
    print("Pending Pods summary")
    print_table(["KIND", "PENDING_PODS"], [
        ["CPU", len(pending_cpu)],
        ["GPU", len(pending_gpu)],
    ])
    # Top reasons (from newest FailedScheduling events used)
    def top_reason_lines(kind, reasons):
        kv = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        lines = []
        for k, v in kv[:6]:
            lines.append([kind, f"{k}", v])
        if not lines:
            lines.append([kind, "(no recent FailedScheduling reasons found)", ""])
        return lines

    reason_rows = []
    reason_rows += top_reason_lines("CPU", cpu_reasons)
    reason_rows += top_reason_lines("GPU", gpu_reasons)
    print_table(["KIND", "REASON (normalized)", "COUNT"], reason_rows)
    print()

    # ResourceQuota (namespace)
    def quota_value(qstatus, key):
        try:
            return qstatus.hard.get(key), qstatus.used.get(key)
        except Exception:
            return None, None

    rq_rows = []
    for rq in quotas:
        st = rq.status
        if not st:
            continue
        # Show a few common fields if present
        for res_key in ["requests.cpu", "limits.cpu", "requests.memory", "limits.memory",
                        "requests.nvidia.com/gpu", "limits.nvidia.com/gpu"]:
            hard, used = quota_value(st, res_key)
            if hard or used:
                rq_rows.append([rq.metadata.name, res_key, used or "-", hard or "-"])
    if rq_rows:
        print("Namespace ResourceQuotas (usage / hard)")
        print_table(["RQ NAME", "RESOURCE", "USED", "HARD"], rq_rows)
        print()
    else:
        print("Namespace ResourceQuotas: none detected or not accessible")
        print()

    # Volcano PodGroups view (helpful to see batch groups & minAvailable)
    if podgroups:
        pg_rows = []
        for pg in podgroups:
            name = pg.get("metadata", {}).get("name", "?")
            status = pg.get("status", {})
            phase = status.get("phase", "-")
            running = status.get("running", 0)
            succeeded = status.get("succeeded", 0)
            failed = status.get("failed", 0)
            min_avail = pg.get("spec", {}).get("minMember", pg.get("spec", {}).get("minAvailable", "-"))
            pg_rows.append([name, phase, min_avail, running, succeeded, failed])
        print("Volcano PodGroups (if present)")
        print_table(["NAME", "PHASE", "MIN_AVAIL", "RUNNING", "SUCCEEDED", "FAILED"], pg_rows)
        print()

    # Helpful footnotes
    print("Notes:")
    print("- Counts under 'Schedulability' and 'Availability...' are inferred from the scheduler's latest")
    print("  FailedScheduling events for your namespace's Pods (no cluster-wide permissions required).")
    print("- 'Full (capacity-limited/absent)' for GPU includes nodes without GPUs or with insufficient GPUs")
    print("  for the pending workload size, as reported by the scheduler.")
    print("- Observed node counts only include nodes where *this namespace's* Pods are currently running.")
    print()
    # Show which events were used
    if cpu_ev or gpu_ev:
        print("Event snapshots used:")
        if cpu_ev:
            print(f"  CPU: {cpu_ev.get('involved_object_name')} at {human_ts(cpu_ev.get('last_timestamp'))}")
        if gpu_ev:
            print(f"  GPU: {gpu_ev.get('involved_object_name')} at {human_ts(gpu_ev.get('last_timestamp'))}")
    else:
        print("No recent FailedScheduling events found for CPU or GPU Pods in this namespace.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
