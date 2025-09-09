#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
volcano_ns_node_stats.py (rev2)

Fixes:
- Issue #1: Avoid unhelpful "N/A" by:
  * Parsing Pod status conditions (PodScheduled=False, reason=Unschedulable) to extract
    "X/Y nodes are available" lines (works even when Events are absent/rotated).
  * Considering both Pod and PodGroup events (Volcano), with broader reasons,
    as long as the message includes "X/Y nodes are available".
  * Falling back to "lower-bound" schedulable counts from observed nodes that already
    run this namespace's Pods (no cluster-wide perms).

- Issue #2: Print currently running Volcano Jobs (vcjobs) and resources they request:
  * Lists batch.volcano.sh Jobs (tries v1alpha1 then v1beta1).
  * Computes desired vs active (running) resource requests per task and per job.
  * Shows: NAME, STATE, QUEUE, AGE, RUNNING/DESIRED, GPU(active/desired), CPU(m),
    MEM(GB), MIN_AVAIL, plus PodGroup status if reachable.

Constraints:
- No metrics-server needed.
- All calls restricted to VOLCANO_NAMESPACE.
"""

import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Iterable

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
        try:
            return int(float(str(x)))
        except Exception:
            return default


def pad(s: str, w: int) -> str:
    return s + (" " * max(0, w - len(s)))


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = "  " + "  ".join(pad(h, widths[i]) for i, h in enumerate(headers))
    sep_line = "  " + "  ".join("-" * widths[i] for i, _ in enumerate(headers))
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  " + "  ".join(pad(str(cell), widths[i]) for i, cell in enumerate(row)))


# ---- Quantity parsing (no external deps) ---- #

_BIN = {
    "ki": 1024**1, "mi": 1024**2, "gi": 1024**3, "ti": 1024**4, "pi": 1024**5, "ei": 1024**6
}
_DEC = {
    "k": 1000**1, "m": 1000**2, "g": 1000**3, "t": 1000**4, "p": 1000**5, "e": 1000**6
}

def parse_cpu_millicores(v: Any) -> int:
    """
    '500m' -> 500
    '1' -> 1000
    '2.5' -> 2500
    """
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s.endswith("m"):
        return safe_int(s[:-1], 0)
    # cores -> m
    try:
        return int(float(s) * 1000.0)
    except Exception:
        return 0

def parse_mem_bytes(v: Any) -> int:
    """
    Supports Ki/Mi/Gi (binary) and K/M/G (decimal), and plain integers (bytes).
    """
    if v is None:
        return 0
    s = str(v).strip()
    try:
        # plain int bytes
        return int(s)
    except Exception:
        pass
    s = s.lower()
    # split number and unit
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit() or ch in ".-+":
            num += ch
        else:
            unit += ch
    try:
        f = float(num)
    except Exception:
        return 0
    unit = unit.strip()
    if unit in _BIN:
        return int(f * _BIN[unit])
    if unit in _DEC:
        return int(f * _DEC[unit])
    # handle like 'gb', 'mb'
    unit = unit.rstrip("b")  # drop optional trailing 'b'
    if unit in _BIN:
        return int(f * _BIN[unit])
    if unit in _DEC:
        return int(f * _DEC[unit])
    # default assume bytes
    return int(f)


def bytes_to_gb(b: int) -> str:
    if b <= 0:
        return "0"
    return f"{b / (1024**3):.2f}"


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
    Volcano PodGroups (namespaced CRD).
    GVR: scheduling.volcano.sh/v1beta1, plural: podgroups
    """
    try:
        api = client.CustomObjectsApi()
        resp = api.list_namespaced_custom_object(
            group="scheduling.volcano.sh", version="v1beta1",
            namespace=ns, plural="podgroups"
        )
        return resp.get("items", []) or []
    except ApiException:
        return []
    except Exception:
        return []


def list_ns_events(ns: str) -> List[dict]:
    """
    Return events (as dicts) for events.k8s.io/v1 and core/v1.
    Normalized fields: reason, message, type, involved_object_name, involved_object_kind, last_timestamp
    """
    out: List[dict] = []

    # events.k8s.io/v1
    try:
        ev1 = client.EventsV1Api()
        resp = ev1.list_namespaced_event(ns)
        for e in resp.items or []:
            reason = getattr(e, "reason", "") or ""
            note = getattr(e, "note", "") or ""
            etype = getattr(e, "type", "") or ""
            regarding = getattr(e, "regarding", None)
            obj_name = getattr(regarding, "name", "") if regarding else ""
            obj_kind = getattr(regarding, "kind", "") if regarding else ""
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
                "involved_object_kind": obj_kind,
                "last_timestamp": ts
            })
    except ApiException:
        pass
    except Exception:
        pass

    # core/v1 events
    try:
        v1 = client.CoreV1Api()
        resp = v1.list_namespaced_event(ns)
        for e in resp.items or []:
            reason = getattr(e, "reason", "") or ""
            msg = getattr(e, "message", "") or ""
            etype = getattr(e, "type", "") or ""
            inv = getattr(e, "involved_object", None)
            obj_name = getattr(inv, "name", "") if inv else ""
            obj_kind = getattr(inv, "kind", "") if inv else ""
            ts = getattr(e, "last_timestamp", None) or getattr(e, "event_time", None)
            out.append({
                "reason": reason,
                "message": msg,
                "type": etype,
                "involved_object_name": obj_name,
                "involved_object_kind": obj_kind,
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
            try:
                return int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
            except Exception:
                return 0

    out.sort(key=ts_key, reverse=True)
    return out


# -------------------------- Parsing & Aggregation -------------------------- #

AVAIL_RE = re.compile(r'(\d+)\s*/\s*(\d+)\s+nodes?\s+(?:are|were)\s+available', re.IGNORECASE)
COUNT_PREFIX_RE = re.compile(r'^\s*(\d+)\s+(.+?)\s*$', re.IGNORECASE)

ALLOWED_UNSCHED_REASONS = {
    "FailedScheduling", "Unschedulable", "NotTriggerScaleUp", "PodGroupNotReady", "EnqueueFailed"
}

def sum_gpu_requests_in_pod(p: client.V1Pod) -> int:
    """Sum requested NVIDIA GPUs across containers/initContainers (requests or limits)."""
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
        total += get_gpu(getattr(c, "resources", None))
    # init containers: take max (they don't run concurrently with app containers)
    for c in (p.spec.init_containers or []):
        total = max(total, get_gpu(getattr(c, "resources", None)))
    return total


def is_gpu_pod(p: client.V1Pod) -> bool:
    return sum_gpu_requests_in_pod(p) > 0


def is_pending(p: client.V1Pod) -> bool:
    return (p.status and p.status.phase == "Pending")


def parse_failed_scheduling_message(msg: str) -> Tuple[Optional[int], Optional[int], Dict[str, int]]:
    """
    Extract (available, total, reasons) from a message with "X/Y nodes are available: N <reason>, ...".
    """
    if not msg:
        return None, None, {}

    available, total = None, None
    m = AVAIL_RE.search(msg)
    if m:
        available = safe_int(m.group(1))
        total = safe_int(m.group(2))

    reasons: Dict[str, int] = {}
    parts = msg.split(":", 1)
    if len(parts) == 2:
        tail = parts[1]
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
            elif "had taint {" in text or "taint" in text:
                key = "taints"
            elif "didn't match node selector" in text or "did not match node selector" in text:
                key = "node_selector_mismatch"
            elif "were unschedulable" in text or "unschedulable" in text:
                key = "unschedulable"
            elif "preemption is not helpful" in text:
                key = "preemption_not_helpful"
            elif "ineligible" in text:
                key = "ineligible"
            else:
                key = f"other/{text[:48]}"
            reasons[key] = reasons.get(key, 0) + count

    return available, total, reasons


def pick_best_from_pod_conditions(pods: List[client.V1Pod]) -> Optional[Tuple[int, int, Dict[str, int], dict]]:
    """
    Find the newest PodScheduled=False, reason=Unschedulable condition that contains "X/Y nodes are available".
    Returns (avail, total, reasons, meta)
    """
    best = None
    for p in pods:
        conds = (p.status.conditions or [])
        for c in conds:
            if getattr(c, "type", "") != "PodScheduled":
                continue
            if getattr(c, "status", "") != "False":
                continue
            reason = (getattr(c, "reason", "") or "")
            if reason not in {"Unschedulable", "PodGroupNotReady"}:
                # still allow if message contains X/Y despite different reason
                pass
            msg = getattr(c, "message", "") or ""
            a, t, r = parse_failed_scheduling_message(msg)
            if a is None or t is None:
                continue
            ts = getattr(c, "last_transition_time", None) or p.metadata.creation_timestamp
            meta = {"source": "pod_condition", "pod": p.metadata.name, "ts": ts}
            if not best:
                best = (a, t, r, meta)
            else:
                # prefer the newest by ts; if equal, prefer larger total (more informative)
                prev_ts = best[3].get("ts")
                prev_ts_int = int(prev_ts.timestamp()) if prev_ts else 0
                ts_int = int(ts.timestamp()) if ts else 0
                if ts_int > prev_ts_int or (ts_int == prev_ts_int and t > best[1]):
                    best = (a, t, r, meta)
    return best


def pick_best_from_events(events: List[dict], names: Iterable[str], kinds: Iterable[str]) -> Optional[Tuple[int, int, Dict[str, int], dict]]:
    """
    Find the newest event for the given object names and kinds whose message includes "X/Y nodes are available".
    Returns (avail, total, reasons, meta)
    """
    name_set = set(names)
    kind_set = set(kinds)
    for e in events:  # already sorted newest-first
        if e.get("involved_object_kind") not in kind_set:
            continue
        if name_set and e.get("involved_object_name") not in name_set:
            continue
        if e.get("reason") not in ALLOWED_UNSCHED_REASONS:
            # Accept other reasons if the message has the X/Y pattern
            pass
        a, t, r = parse_failed_scheduling_message(e.get("message", "") or "")
        if a is None or t is None:
            continue
        return a, t, r, {
            "source": "event",
            "obj_kind": e.get("involved_object_kind"),
            "obj_name": e.get("involved_object_name"),
            "ts": e.get("last_timestamp")
        }
    return None


def aggregate_schedulability_for_kind(
    pods: List[client.V1Pod],
    events: List[dict],
    podgroups: List[dict],
    kind: str  # "CPU" or "GPU"
) -> Tuple[Optional[int], Optional[int], Dict[str, int], Optional[dict]]:
    """
    Return (available, total, reasons, meta) for the given kind.
    Tries pod conditions first, then events for Pods, then events for PodGroups tied to those Pods.
    """
    if kind not in ("CPU", "GPU"):
        raise ValueError("kind must be CPU or GPU")

    if kind == "GPU":
        kind_pods = [p for p in pods if is_gpu_pod(p)]
    else:
        kind_pods = [p for p in pods if not is_gpu_pod(p)]

    # 1) Pod conditions (best when events are rotated/missing)
    best_cond = pick_best_from_pod_conditions([p for p in kind_pods if is_pending(p)])
    if not best_cond:
        # also consider any pods of that kind (even if not Pending) to catch recent transitions
        best_cond = pick_best_from_pod_conditions(kind_pods)
    if best_cond:
        a, t, r, meta = best_cond
        return a, t, r, meta

    # 2) Events for Pods
    kind_pod_names = [p.metadata.name for p in kind_pods]
    best_evt = pick_best_from_events(events, kind_pod_names, kinds={"Pod"})
    if best_evt:
        return best_evt

    # 3) Events for PodGroups associated with our Pods
    # Map PG -> whether it hosts any GPU pods
    # Volcano labels: pods usually have label "volcano.sh/job-name" and/or "volcano.sh/podgroup"
    pg_to_kind = {}  # name -> "GPU"/"CPU"
    # from pods
    for p in kind_pods:
        labels = p.metadata.labels or {}
        pg = labels.get("volcano.sh/podgroup") or labels.get("volcano.sh/pod-group") or ""
        if pg:
            pg_to_kind[pg] = kind
    # from podgroups themselves (if spec or status links exist we ignore; name alone is fine)
    pg_names = [pg.get("metadata", {}).get("name", "") for pg in podgroups]
    # Consider all PGs for this kind if we saw any mapping; otherwise none
    if pg_to_kind:
        target_pg_names = [name for name, k in pg_to_kind.items() if k == kind]
        best_pg_evt = pick_best_from_events(events, target_pg_names, kinds={"PodGroup"})
        if best_pg_evt:
            return best_pg_evt

    return None, None, {}, None


# ----------------------- Volcano Jobs (vcjob) reporting -------------------- #

def list_ns_vcjobs(ns: str) -> List[dict]:
    """
    List Volcano Jobs in namespace. Try v1alpha1 then v1beta1.
    Returns raw dicts.
    """
    api = client.CustomObjectsApi()
    for ver in ("v1alpha1", "v1beta1"):
        try:
            resp = api.list_namespaced_custom_object(
                group="batch.volcano.sh", version=ver, namespace=ns, plural="jobs"
            )
            items = resp.get("items", []) or []
            # annotate version used
            for it in items:
                it["_apiVersionResolved"] = f"batch.volcano.sh/{ver}"
            return items
        except ApiException as e:
            if e.status in (403, 404):
                continue
    return []


def per_pod_requests_from_task(task: dict) -> Tuple[int, int, int]:
    """
    Returns (gpu, cpu_m, mem_bytes) requested by a single pod of this task.
    """
    gpu = 0
    cpu_m = 0
    mem_b = 0

    tpl = (task.get("template") or {}).get("spec", {})  # PodSpec
    containers = tpl.get("containers", []) or []
    init_containers = tpl.get("initContainers", []) or []

    def add_from_res(res: dict):
        nonlocal gpu, cpu_m, mem_b
        if not res:
            return
        req = res.get("requests", {}) or {}
        lim = res.get("limits", {}) or {}
        # Prefer requests, but if not present, use limits for GPU
        g = req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu"))
        if g is not None:
            try:
                gpu += int(g)
            except Exception:
                pass
        c = req.get("cpu")
        if c is not None:
            cpu_m += parse_cpu_millicores(c)
        m = req.get("memory")
        if m is not None:
            mem_b += parse_mem_bytes(m)

    for c in containers:
        add_from_res((c.get("resources") or {}))
    # init containers: take max, not sum, since they do not run concurrently
    init_cpu_m = 0
    init_mem_b = 0
    init_gpu = 0
    for c in init_containers:
        res = (c.get("resources") or {})
        req = res.get("requests", {}) or {}
        lim = res.get("limits", {}) or {}
        g = req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu"))
        if g is not None:
            try:
                init_gpu = max(init_gpu, int(g))
            except Exception:
                pass
        cval = req.get("cpu")
        if cval is not None:
            init_cpu_m = max(init_cpu_m, parse_cpu_millicores(cval))
        mval = req.get("memory")
        if mval is not None:
            init_mem_b = max(init_mem_b, parse_mem_bytes(mval))
    gpu = max(gpu, init_gpu)
    cpu_m = max(cpu_m, init_cpu_m)
    mem_b = max(mem_b, init_mem_b)

    return gpu, cpu_m, mem_b


def summarize_vcjobs(ns: str, pods: List[client.V1Pod]) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Returns two tables (rows arrays):
      running_rows, other_rows
    """
    jobs = list_ns_vcjobs(ns)
    if not jobs:
        return [], []

    # Build helper index of pods by job and by task
    # Volcano labels typically:
    #   "volcano.sh/job-name": <job>
    #   "volcano.sh/task-name": <task-name>
    pods_by_job: Dict[str, List[client.V1Pod]] = {}
    pods_by_job_task: Dict[Tuple[str, str], List[client.V1Pod]] = {}
    for p in pods:
        labels = p.metadata.labels or {}
        jn = labels.get("volcano.sh/job-name") or labels.get("volcano.sh/job") or ""
        tn = labels.get("volcano.sh/task-name") or labels.get("volcano.sh/task") or ""
        if jn:
            pods_by_job.setdefault(jn, []).append(p)
            if tn:
                pods_by_job_task.setdefault((jn, tn), []).append(p)

    running_rows = []
    other_rows = []

    for j in jobs:
        meta = j.get("metadata", {}) or {}
        spec = j.get("spec", {}) or {}
        status = j.get("status", {}) or {}

        name = meta.get("name", "?")
        queue = spec.get("queue", "-")
        min_avail = spec.get("minAvailable", "-")
        phase = status.get("state") or status.get("phase") or "-"
        created = meta.get("creationTimestamp")
        try:
            age_dt = datetime.now(timezone.utc) - datetime.fromisoformat(created.replace("Z", "+00:00"))
            age = f"{int(age_dt.total_seconds() // 3600)}h"
        except Exception:
            age = "-"

        # tasks
        tasks = spec.get("tasks", []) or []
        desired_total = 0
        desired_gpu = 0
        desired_cpu_m = 0
        desired_mem_b = 0

        active_total = 0
        active_gpu = 0
        active_cpu_m = 0
        active_mem_b = 0

        # PodGroup status if available
        pg_status = status.get("podGroup", {}) or {}
        pg_running = pg_status.get("running", None)
        pg_succeeded = pg_status.get("succeeded", None)
        pg_failed = pg_status.get("failed", None)

        for t in tasks:
            tname = t.get("name", "")
            replicas = safe_int(t.get("replicas", 0), 0)
            per_gpu, per_cpu_m, per_mem_b = per_pod_requests_from_task(t)

            desired_total += replicas
            desired_gpu += per_gpu * replicas
            desired_cpu_m += per_cpu_m * replicas
            desired_mem_b += per_mem_b * replicas

            # running pods for this task (by label)
            running_pods = [p for p in pods_by_job_task.get((name, tname), []) if (p.status and p.status.phase == "Running")]
            rcount = len(running_pods)

            active_total += rcount
            active_gpu += per_gpu * rcount
            active_cpu_m += per_cpu_m * rcount
            active_mem_b += per_mem_b * rcount

        row = [
            name,
            phase,
            queue or "-",
            age,
            f"{active_total}/{desired_total}",
            f"{active_gpu}/{desired_gpu}",
            f"{active_cpu_m}/{desired_cpu_m}",
            f"{bytes_to_gb(active_mem_b)}/{bytes_to_gb(desired_mem_b)}",
            min_avail,
            (pg_running if pg_running is not None else "-"),
            (pg_succeeded if pg_succeeded is not None else "-"),
            (pg_failed if pg_failed is not None else "-"),
            j.get("_apiVersionResolved", "-")
        ]
        if str(phase).lower() == "running":
            running_rows.append(row)
        else:
            other_rows.append(row)

    return running_rows, other_rows


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
    cpu_avail, cpu_total, cpu_reasons, cpu_meta = aggregate_schedulability_for_kind(pods, events, podgroups, "CPU")
    gpu_avail, gpu_total, gpu_reasons, gpu_meta = aggregate_schedulability_for_kind(pods, events, podgroups, "GPU")

    # Fallback lower bounds if nothing found
    observed_gpu_nodes = sum(1 for n, rec in observed_nodes.items() if rec["has_gpu_pod"])
    observed_cpu_nodes = len(observed_nodes) - observed_gpu_nodes

    print(f"Namespace: {ns}")
    print(f"Generated at: {ts}")
    print()

    # ------------------- Schedulability (node counts) ------------------- #
    print("Schedulability (node counts)")
    sched_rows = []
    def add_sched(kind, avail, total, observed_lb):
        if avail is not None and total is not None:
            sched_rows.append([kind, "Schedulable", avail])
            sched_rows.append([kind, "Unschedulable", total - avail])
        else:
            # No direct diagnostics -> show lower bound instead of N/A
            sched_rows.append([kind, "Schedulable (lower bound)", f">= {observed_lb}"])
            sched_rows.append([kind, "Unschedulable (unknown)", "-"])

    add_sched("CPU", cpu_avail, cpu_total, observed_cpu_nodes)
    add_sched("GPU", gpu_avail, gpu_total, observed_gpu_nodes)

    print_table(["KIND", "SCHED", "COUNT"], sched_rows)
    print()

    # ---------- Availability among Schedulable worker nodes ------------- #
    print("Availability among Schedulable worker nodes (node counts)")
    avail_rows = []
    def breakdown(avail, total, reasons, kind):
        if avail is None or total is None:
            return [[kind, "Available (lower bound)", f">= {observed_gpu_nodes if kind=='GPU' else observed_cpu_nodes}"],
                    [kind, "Full (capacity-limited)", "-"] if kind=="CPU" else
                    [kind, "Full (capacity-limited/absent)", "-"],
                    [kind, "Other unschedulable reasons", "-"]]
        # "Full" reasons
        if kind == "GPU":
            full = reasons.get("insufficient/nvidia.com/gpu", 0)
            full_label = "Full (capacity-limited/absent)"
        else:
            full = reasons.get("insufficient/cpu", 0)
            full_label = "Full (capacity-limited)"
        other = max(0, (total - avail) - full)
        return [[kind, "Available", avail],
                [kind, full_label, full],
                [kind, "Other unschedulable reasons", other]]

    avail_rows += breakdown(cpu_avail, cpu_total, cpu_reasons, "CPU")
    avail_rows += breakdown(gpu_avail, gpu_total, gpu_reasons, "GPU")
    print_table(["KIND", "STATE", "COUNT"], avail_rows)
    print()

    # ----------------------- Observed nodes summary ---------------------- #
    total_observed = len(observed_nodes)
    print("Observed nodes currently running this namespace's Pods")
    print_table(["METRIC", "VALUE"], [
        ["Distinct nodes observed", total_observed],
        ["Nodes observed with GPU Pods (ns)", observed_gpu_nodes],
        ["Nodes observed with only CPU Pods (ns)", observed_cpu_nodes],
    ])
    print()

    # ------------------------ Pending Pods summary ----------------------- #
    pending_gpu = [p for p in pods if is_gpu_pod(p) and is_pending(p)]
    pending_cpu = [p for p in pods if (not is_gpu_pod(p)) and is_pending(p)]
    print("Pending Pods summary")
    print_table(["KIND", "PENDING_PODS"], [
        ["CPU", len(pending_cpu)],
        ["GPU", len(pending_gpu)],
    ])
    # Top reasons (from whichever diagnostics were used)
    def top_reason_lines(kind, reasons):
        kv = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        lines = []
        for k, v in kv[:6]:
            lines.append([kind, f"{k}", v])
        if not lines:
            lines.append([kind, "(no recent unschedulable diagnostics found)", ""])
        return lines

    reason_rows = []
    reason_rows += top_reason_lines("CPU", cpu_reasons)
    reason_rows += top_reason_lines("GPU", gpu_reasons)
    print_table(["KIND", "REASON (normalized)", "COUNT"], reason_rows)
    print()

    # ---------------------- Namespace ResourceQuotas --------------------- #
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

    # ---------------------- Volcano Jobs (vcjob) ------------------------- #
    running_rows, other_rows = summarize_vcjobs(ns, pods)

    print("Running Volcano Jobs (requested resources)")
    if running_rows:
        print_table(
            ["NAME", "STATE", "QUEUE", "AGE", "RUNNING/DESIRED", "GPU (act/des)", "CPU m (act/des)", "MEM Gi (act/des)", "MIN_AVAIL", "PG RUN", "PG SUCC", "PG FAIL", "API"],
            running_rows
        )
    else:
        print("  (none)")
    print()

    print("Other Volcano Jobs")
    if other_rows:
        print_table(
            ["NAME", "STATE", "QUEUE", "AGE", "RUNNING/DESIRED", "GPU (act/des)", "CPU m (act/des)", "MEM Gi (act/des)", "MIN_AVAIL", "PG RUN", "PG SUCC", "PG FAIL", "API"],
            other_rows
        )
    else:
        print("  (none)")
    print()

    # ---------------------------- Footnotes ------------------------------ #
    print("Notes:")
    print("- Primary diagnostics come from PodScheduled=False 'Unschedulable' condition messages and/or Events")
    print("  that include 'X/Y nodes are available: ...'. These are namespace-scoped and require no node list.")
    print("- When such diagnostics were not found, 'lower bound' counts reflect nodes already hosting this")
    print("  namespace's Pods (observed), which is conservative but avoids unhelpful 'N/A'.")
    print("- Volcano Job resources are computed from task Pod templates (requested resources),")
    print("  and 'active' requests are based on currently running pods per task.")
    print()

    # Show which diagnostic source was used
    if cpu_meta or gpu_meta:
        print("Diagnostics source:")
        if cpu_meta:
            src = cpu_meta.get("source")
            ident = cpu_meta.get("pod") or cpu_meta.get("obj_name") or "-"
            print(f"  CPU: {src} -> {ident} at {human_ts(cpu_meta.get('ts'))}")
        else:
            print("  CPU: lower-bound fallback from observed nodes")
        if gpu_meta:
            src = gpu_meta.get("source")
            ident = gpu_meta.get("pod") or gpu_meta.get("obj_name") or "-"
            print(f"  GPU: {src} -> {ident} at {human_ts(gpu_meta.get('ts'))}")
        else:
            print("  GPU: lower-bound fallback from observed nodes")
    else:
        print("Diagnostics source: lower-bound fallback from observed nodes (no recent unschedulable diagnostics found)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
