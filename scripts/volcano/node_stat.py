#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
volcano_ns_node_stats.py (rev3)

Fixes for user-reported issues:
1) Eliminate 'N/A' in schedulability/availability tables. Use concrete numbers when available,
   otherwise '>= <lower-bound>' and 'unknown' (never 'N/A').
2) Show ONLY Volcano Jobs that are in phase/state 'Running'.
3) Clarify 'GPU (act/des)' meaning in notes.

Assumptions/constraints:
- No cluster-wide permissions. Namespace is provided via VOLCANO_NAMESPACE env var.
- No metrics-server. We infer from Pod conditions and Events (namespaced).
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

def human_ts(ts) -> str:
    try:
        if not ts:
            return "unknown"
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "unknown"

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
    print("  " + "  ".join(pad(h, widths[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(pad(str(cell), widths[i]) for i, cell in enumerate(row)))

# ---- Quantity parsing ---- #

_BIN = {"ki": 1024**1, "mi": 1024**2, "gi": 1024**3, "ti": 1024**4, "pi": 1024**5, "ei": 1024**6}
_DEC = {"k": 1000**1, "m": 1000**2, "g": 1000**3, "t": 1000**4, "p": 1000**5, "e": 1000**6}

def parse_cpu_millicores(v: Any) -> int:
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s.endswith("m"):
        return safe_int(s[:-1], 0)
    try:
        return int(float(s) * 1000.0)
    except Exception:
        return 0

def parse_mem_bytes(v: Any) -> int:
    if v is None:
        return 0
    s = str(v).strip()
    try:
        return int(s)
    except Exception:
        pass
    s = s.lower()
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
    u = unit.strip().rstrip("b")
    if u in _BIN:
        return int(f * _BIN[u])
    if u in _DEC:
        return int(f * _DEC[u])
    return int(f)

def bytes_to_gib(b: int) -> str:
    if b <= 0:
        return "0.00"
    return f"{b / (1024**3):.2f}"

# ----------------------------- Kubernetes I/O ------------------------------ #

def list_ns_pods(v1: client.CoreV1Api, ns: str) -> List[client.V1Pod]:
    return (v1.list_namespaced_pod(ns).items or [])

def list_ns_resourcequotas(v1: client.CoreV1Api, ns: str) -> List[client.V1ResourceQuota]:
    try:
        return (v1.list_namespaced_resource_quota(ns).items or [])
    except ApiException as e:
        if e.status == 403:
            return []
        raise

def list_ns_podgroups(ns: str) -> List[dict]:
    try:
        api = client.CustomObjectsApi()
        resp = api.list_namespaced_custom_object(
            group="scheduling.volcano.sh", version="v1beta1", namespace=ns, plural="podgroups"
        )
        return resp.get("items", []) or []
    except ApiException:
        return []
    except Exception:
        return []

def list_ns_events(ns: str) -> List[dict]:
    out: List[dict] = []
    # events.k8s.io/v1
    try:
        ev1 = client.EventsV1Api()
        resp = ev1.list_namespaced_event(ns)
        for e in resp.items or []:
            out.append({
                "reason": getattr(e, "reason", "") or "",
                "message": getattr(e, "note", "") or "",
                "type": getattr(e, "type", "") or "",
                "obj_name": getattr(getattr(e, "regarding", None), "name", "") if getattr(e, "regarding", None) else "",
                "obj_kind": getattr(getattr(e, "regarding", None), "kind", "") if getattr(e, "regarding", None) else "",
                "ts": getattr(e, "event_time", None) or (getattr(getattr(e, "series", None), "last_observed_time", None)) or getattr(e, "deprecated_last_timestamp", None),
            })
    except Exception:
        pass
    # core/v1
    try:
        v1 = client.CoreV1Api()
        resp = v1.list_namespaced_event(ns)
        for e in resp.items or []:
            out.append({
                "reason": getattr(e, "reason", "") or "",
                "message": getattr(e, "message", "") or "",
                "type": getattr(e, "type", "") or "",
                "obj_name": getattr(getattr(e, "involved_object", None), "name", "") if getattr(e, "involved_object", None) else "",
                "obj_kind": getattr(getattr(e, "involved_object", None), "kind", "") if getattr(e, "involved_object", None) else "",
                "ts": getattr(e, "last_timestamp", None) or getattr(e, "event_time", None),
            })
    except Exception:
        pass

    def ts_key(d: dict):
        t = d.get("ts")
        if not t:
            return 0
        try:
            return int(t.timestamp())  # datetime
        except Exception:
            try:
                return int(datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp())
            except Exception:
                return 0

    out.sort(key=ts_key, reverse=True)
    return out

def list_ns_vcjobs(ns: str) -> List[dict]:
    api = client.CustomObjectsApi()
    for ver in ("v1alpha1", "v1beta1"):
        try:
            resp = api.list_namespaced_custom_object(
                group="batch.volcano.sh", version=ver, namespace=ns, plural="jobs"
            )
            items = resp.get("items", []) or []
            for it in items:
                it["_apiVersionResolved"] = f"batch.volcano.sh/{ver}"
            return items
        except ApiException as e:
            if e.status in (403, 404):
                continue
    return []

# -------------------------- Parsing & Aggregation -------------------------- #

AVAIL_RE = re.compile(r'(\d+)\s*/\s*(\d+)\s+nodes?\s+(?:are|were)\s+available', re.IGNORECASE)
COUNT_PREFIX_RE = re.compile(r'^\s*(\d+)\s+(.+?)\s*$', re.IGNORECASE)
UNSCHED_REASONS_ALLOW = {"FailedScheduling", "Unschedulable", "NotTriggerScaleUp", "PodGroupNotReady", "EnqueueFailed"}

def sum_gpu_requests_in_pod(p: client.V1Pod) -> int:
    def get_gpu(res):
        if not res:
            return 0
        req = 0
        if res.requests and "nvidia.com/gpu" in res.requests:
            try: req = max(req, int(res.requests["nvidia.com/gpu"]))
            except Exception: pass
        if res.limits and "nvidia.com/gpu" in res.limits:
            try: req = max(req, int(res.limits["nvidia.com/gpu"]))
            except Exception: pass
        return req

    total = 0
    for c in (p.spec.containers or []):
        total += get_gpu(getattr(c, "resources", None))
    # init: not concurrent with app; take max
    init_max = 0
    for c in (p.spec.init_containers or []):
        init_max = max(init_max, get_gpu(getattr(c, "resources", None)))
    return max(total, init_max)

def is_gpu_pod(p: client.V1Pod) -> bool:
    return sum_gpu_requests_in_pod(p) > 0

def is_pending(p: client.V1Pod) -> bool:
    return (p.status and p.status.phase == "Pending")

def parse_failed_scheduling_message(msg: str) -> Tuple[Optional[int], Optional[int], Dict[str, int]]:
    if not msg:
        return None, None, {}
    a, t = None, None
    m = AVAIL_RE.search(msg)
    if m:
        a = safe_int(m.group(1))
        t = safe_int(m.group(2))
    reasons: Dict[str, int] = {}
    parts = msg.split(":", 1)
    if len(parts) == 2:
        tail = parts[1]
        frags = [x.strip(" .") for x in tail.split(",") if x.strip()]
        for f in frags:
            m2 = COUNT_PREFIX_RE.match(f)
            if not m2:
                continue
            cnt = safe_int(m2.group(1))
            txt = m2.group(2).lower()
            if txt.startswith("insufficient "):
                key = f"insufficient/{txt.split(' ', 1)[1]}"
            elif "taint" in txt:
                key = "taints"
            elif "node selector" in txt:
                key = "node_selector_mismatch"
            elif "unschedulable" in txt:
                key = "unschedulable"
            elif "preemption is not helpful" in txt:
                key = "preemption_not_helpful"
            else:
                key = f"other/{txt[:48]}"
            reasons[key] = reasons.get(key, 0) + cnt
    return a, t, reasons

def collect_diag_from_pod_conditions(pods: List[client.V1Pod]) -> List[Tuple[int,int,Dict[str,int],dict]]:
    out = []
    for p in pods:
        conds = (p.status.conditions or [])
        for c in conds:
            if getattr(c, "type", "") != "PodScheduled":
                continue
            if getattr(c, "status", "") != "False":
                continue
            msg = getattr(c, "message", "") or ""
            a, t, r = parse_failed_scheduling_message(msg)
            if a is None or t is None:
                continue
            ts = getattr(c, "last_transition_time", None) or p.metadata.creation_timestamp
            out.append((a, t, r, {"src":"pod_condition", "pod":p.metadata.name, "ts":ts}))
    # newest first
    out.sort(key=lambda x: int(x[3]["ts"].timestamp()) if x[3].get("ts") else 0, reverse=True)
    return out

def collect_diag_from_events(events: List[dict], names: Iterable[str], kinds: Iterable[str]) -> List[Tuple[int,int,Dict[str,int],dict]]:
    name_set = set(names)
    kind_set = set(kinds)
    out = []
    for e in events:  # already newest-first
        if kind_set and e.get("obj_kind") not in kind_set:
            continue
        if name_set and e.get("obj_name") not in name_set:
            continue
        # accept if message has the pattern, regardless of reason
        if e.get("reason") not in UNSCHED_REASONS_ALLOW and AVAIL_RE.search(e.get("message","")) is None:
            continue
        a, t, r = parse_failed_scheduling_message(e.get("message","") or "")
        if a is None or t is None:
            continue
        out.append((a, t, r, {"src":"event", "obj_kind":e.get("obj_kind"), "obj":e.get("obj_name"), "ts":e.get("ts")}))
    return out

def aggregate_schedulability_for_kind(pods: List[client.V1Pod], events: List[dict], podgroups: List[dict], kind: str):
    """Return (avail,total,reasons,meta, reasons_agg_any_messages) for CPU/GPU."""
    if kind == "GPU":
        kind_pods = [p for p in pods if is_gpu_pod(p)]
    else:
        kind_pods = [p for p in pods if not is_gpu_pod(p)]

    # Collect diagnostics from pod conditions (prefer Pending first)
    pending = [p for p in kind_pods if is_pending(p)]
    diags = collect_diag_from_pod_conditions(pending)
    if not diags:
        diags = collect_diag_from_pod_conditions(kind_pods)

    # Events: Pods
    pod_names = [p.metadata.name for p in kind_pods]
    diags += collect_diag_from_events(events, pod_names, kinds={"Pod"})

    # Events: PodGroups linked to our pods
    pg_names = set()
    for p in kind_pods:
        labels = p.metadata.labels or {}
        pg = labels.get("volcano.sh/podgroup") or labels.get("volcano.sh/pod-group")
        if pg:
            pg_names.add(pg)
    if not pg_names:
        # add PGs present in ns as a last resort
        pg_names = {pg.get("metadata",{}).get("name","") for pg in podgroups if pg.get("metadata")}
    if pg_names:
        diags += collect_diag_from_events(events, pg_names, kinds={"PodGroup"})

    # newest first already since events are newest-first; sort again with best 'ts'
    def ts_val(meta):
        t = meta.get("ts")
        if not t:
            return 0
        try:
            return int(t.timestamp())
        except Exception:
            try:
                return int(datetime.fromisoformat(str(t).replace("Z","+00:00")).timestamp())
            except Exception:
                return 0
    diags.sort(key=lambda x: ts_val(x[3]), reverse=True)

    # Aggregate reason counts from any messages we saw (even if we can't pick a single 'X/Y')
    reasons_agg: Dict[str,int] = {}
    for a,t,r,meta in diags:
        for k,v in r.items():
            reasons_agg[k] = reasons_agg.get(k,0) + v

    if diags:
        # pick the newest diagnostic that has X/Y
        a,t,r,meta = diags[0]
        return a,t,r,meta,reasons_agg

    return None,None,{},None,reasons_agg  # no direct X/Y, but we might still have reasons_agg=={}

# ----------------------- Volcano Jobs (Running only) ----------------------- #

def per_pod_requests_from_task(task: dict) -> Tuple[int,int,int]:
    """(gpu, cpu_m, mem_b) for one pod of this task."""
    gpu = 0; cpu_m = 0; mem_b = 0
    tpl = (task.get("template") or {}).get("spec", {})
    containers = tpl.get("containers", []) or []
    init_containers = tpl.get("initContainers", []) or []

    def add_res(res: dict):
        nonlocal gpu,cpu_m,mem_b
        if not res:
            return
        req = res.get("requests", {}) or {}
        lim = res.get("limits", {}) or {}
        g = req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu"))
        if g is not None:
            try: gpu += int(g)
            except Exception: pass
        c = req.get("cpu")
        if c is not None: cpu_m += parse_cpu_millicores(c)
        m = req.get("memory")
        if m is not None: mem_b += parse_mem_bytes(m)

    for c in containers:
        add_res(c.get("resources") or {})

    # init containers => max (not concurrent)
    init_gpu = 0; init_cpu_m = 0; init_mem_b = 0
    for c in init_containers:
        res = c.get("resources") or {}
        req = res.get("requests", {}) or {}
        lim = res.get("limits", {}) or {}
        g = req.get("nvidia.com/gpu", lim.get("nvidia.com/gpu"))
        if g is not None:
            try: init_gpu = max(init_gpu, int(g))
            except Exception: pass
        cv = req.get("cpu")
        if cv is not None: init_cpu_m = max(init_cpu_m, parse_cpu_millicores(cv))
        mv = req.get("memory")
        if mv is not None: init_mem_b = max(init_mem_b, parse_mem_bytes(mv))
    gpu = max(gpu, init_gpu)
    cpu_m = max(cpu_m, init_cpu_m)
    mem_b = max(mem_b, init_mem_b)
    return gpu,cpu_m,mem_b

def summarize_running_vcjobs(ns: str, pods: List[client.V1Pod]) -> List[List[Any]]:
    jobs = list_ns_vcjobs(ns)
    if not jobs:
        return []

    # index pods by (job, task)
    pods_by_job_task: Dict[Tuple[str,str], List[client.V1Pod]] = {}
    for p in pods:
        labels = p.metadata.labels or {}
        jn = labels.get("volcano.sh/job-name") or labels.get("volcano.sh/job") or ""
        tn = labels.get("volcano.sh/task-name") or labels.get("volcano.sh/task") or ""
        if jn and tn:
            pods_by_job_task.setdefault((jn,tn), []).append(p)

    rows: List[List[Any]] = []
    for j in jobs:
        meta = j.get("metadata", {}) or {}
        spec = j.get("spec", {}) or {}
        status = j.get("status", {}) or {}

        # phase/state may be 'state' or 'phase'
        phase = (status.get("state") or status.get("phase") or "-")
        if str(phase).lower() != "running":
            continue  # show only running

        name = meta.get("name", "?")
        queue = spec.get("queue", "-")
        min_avail = spec.get("minAvailable", "-")
        created = meta.get("creationTimestamp")
        try:
            age = f"{int((datetime.now(timezone.utc) - datetime.fromisoformat(created.replace('Z','+00:00'))).total_seconds()//3600)}h"
        except Exception:
            age = "-"

        tasks = spec.get("tasks", []) or []
        desired_total = 0; desired_gpu = 0; desired_cpu_m = 0; desired_mem_b = 0
        active_total = 0; active_gpu = 0; active_cpu_m = 0; active_mem_b = 0

        pg_status = status.get("podGroup", {}) or {}
        pg_running = pg_status.get("running", "-")
        pg_succeeded = pg_status.get("succeeded", "-")
        pg_failed = pg_status.get("failed", "-")

        for t in tasks:
            tname = t.get("name","")
            replicas = safe_int(t.get("replicas",0),0)
            per_gpu, per_cpu_m, per_mem_b = per_pod_requests_from_task(t)

            desired_total += replicas
            desired_gpu  += per_gpu   * replicas
            desired_cpu_m+= per_cpu_m * replicas
            desired_mem_b+= per_mem_b * replicas

            # active (Running) pods for this task
            running_pods = [p for p in pods_by_job_task.get((name,tname), []) if (p.status and p.status.phase=="Running")]
            rcount = len(running_pods)
            active_total += rcount
            active_gpu   += per_gpu   * rcount
            active_cpu_m += per_cpu_m * rcount
            active_mem_b += per_mem_b * rcount

        rows.append([
            name,
            phase,
            queue or "-",
            age,
            f"{active_total}/{desired_total}",
            f"{active_gpu}/{desired_gpu}",
            f"{active_cpu_m}/{desired_cpu_m}",
            f"{bytes_to_gib(active_mem_b)}/{bytes_to_gib(desired_mem_b)}",
            min_avail,
            pg_running, pg_succeeded, pg_failed,
            j.get("_apiVersionResolved","-")
        ])
    return rows

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
    print(f"Namespace: {ns}")
    print(f"Generated at: {ts}\n")

    # Observed nodes from ns pods
    observed_nodes: Dict[str, Dict[str, Any]] = {}
    for p in pods:
        node = (p.spec.node_name or "").strip()
        if not node:
            continue
        rec = observed_nodes.setdefault(node, {"gpu": False})
        if is_gpu_pod(p):
            rec["gpu"] = True
    observed_gpu_nodes = sum(1 for _,rec in observed_nodes.items() if rec["gpu"])
    observed_cpu_nodes = len(observed_nodes) - observed_gpu_nodes

    # Diagnostics per kind
    cpu_avail, cpu_total, cpu_reasons, cpu_meta, cpu_reasons_any = aggregate_schedulability_for_kind(pods, events, podgroups, "CPU")
    gpu_avail, gpu_total, gpu_reasons, gpu_meta, gpu_reasons_any = aggregate_schedulability_for_kind(pods, events, podgroups, "GPU")

    # ------------------- Schedulability (node counts) ------------------- #
    print("Schedulability (node counts)")
    sched_rows = []

    def add_sched(kind, avail, total, observed_lb):
        if avail is not None and total is not None:
            sched_rows.append([kind, "Schedulable", avail])
            sched_rows.append([kind, "Unschedulable", max(0, total - avail)])
        else:
            # No direct X/Y -> Avoid N/A entirely
            sched_rows.append([kind, "Schedulable (lower bound)", f">= {observed_lb}"])
            sched_rows.append([kind, "Unschedulable (unknown)", "unknown"])

    add_sched("CPU", cpu_avail, cpu_total, observed_cpu_nodes)
    add_sched("GPU", gpu_avail, gpu_total, observed_gpu_nodes)
    print_table(["KIND", "SCHED", "COUNT"], sched_rows)
    print()

    # ---------- Availability among Schedulable worker nodes ------------- #
    print("Availability among Schedulable worker nodes (node counts)")
    avail_rows = []

    def breakdown(kind, avail, total, reasons, reasons_any, lb):
        if avail is None or total is None:
            # When we can't get X/Y, still avoid 'N/A' and show something helpful
            full_guess = (reasons_any.get("insufficient/nvidia.com/gpu", 0) if kind=="GPU"
                          else reasons_any.get("insufficient/cpu", 0))
            label_full = "Full (capacity-limited/absent)" if kind=="GPU" else "Full (capacity-limited)"
            return [[kind, "Available (lower bound)", f">= {lb}"],
                    [kind, label_full, full_guess if full_guess>0 else "unknown"],
                    [kind, "Other unschedulable reasons", "unknown"]]
        # With X/Y available
        full = reasons.get("insufficient/nvidia.com/gpu", 0) if kind=="GPU" else reasons.get("insufficient/cpu", 0)
        other = max(0, (total - avail) - full)
        label_full = "Full (capacity-limited/absent)" if kind=="GPU" else "Full (capacity-limited)"
        return [[kind, "Available", avail],
                [kind, label_full, full],
                [kind, "Other unschedulable reasons", other]]

    avail_rows += breakdown("CPU", cpu_avail, cpu_total, cpu_reasons, cpu_reasons_any, observed_cpu_nodes)
    avail_rows += breakdown("GPU", gpu_avail, gpu_total, gpu_reasons, gpu_reasons_any, observed_gpu_nodes)
    print_table(["KIND", "STATE", "COUNT"], avail_rows)
    print()

    # ----------------------- Observed nodes summary ---------------------- #
    print("Observed nodes currently running this namespace's Pods")
    print_table(["METRIC", "VALUE"], [
        ["Distinct nodes observed", len(observed_nodes)],
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
    print()

    # ---------------------- Namespace ResourceQuotas --------------------- #
    def quota_value(qstatus, key):
        try: return qstatus.hard.get(key), qstatus.used.get(key)
        except Exception: return None, None

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
        print("Namespace ResourceQuotas: none detected or not accessible\n")

    # ---------------------- Volcano Jobs (Running only) ------------------ #
    running_rows = summarize_running_vcjobs(ns, pods)
    print("Running Volcano Jobs (requested resources)")
    if running_rows:
        print_table(
            ["NAME", "STATE", "QUEUE", "AGE", "RUNNING/DESIRED", "GPU (act/des)", "CPU m (act/des)", "MEM Gi (act/des)", "MIN_AVAIL", "PG RUN", "PG SUCC", "PG FAIL", "API"],
            running_rows
        )
    else:
        print("  (none)")
    print()

    # ---------------------------- Notes --------------------------------- #
    print("Notes:")
    print("- Schedulability counts come from the most recent scheduler diagnostics we can access in-namespace:")
    print("  PodScheduled=False condition messages and/or Events that contain 'X/Y nodes are available: ...'.")
    print("- If such diagnostics aren't present, we show conservative *lower bounds* based on nodes already")
    print("  running Pods from this namespace (observed). 'Unknown' means the value cannot be derived")
    print("  without cluster-wide/node permissions and no diagnostics were available.")
    print("- 'GPU (act/des)' for a running vcjob means: the sum of *requested* GPUs across pods that are")
    print("  currently Running in that job (act) / the sum of *requested* GPUs if all desired replicas were running (des).")
    print("  CPU is in millicores; MEM is in GiB, both computed from per-task pod template requests.")
    if cpu_meta or gpu_meta:
        print("- Diagnostics source used:")
        if cpu_meta:
            ident = cpu_meta.get("pod") or cpu_meta.get("obj") or "-"
            print(f"  CPU -> {cpu_meta.get('src')} on {ident} at {human_ts(cpu_meta.get('ts'))}")
        else:
            print("  CPU -> lower-bound fallback (no recent diagnostics)")
        if gpu_meta:
            ident = gpu_meta.get("pod") or gpu_meta.get("obj") or "-"
            print(f"  GPU -> {gpu_meta.get('src')} on {ident} at {human_ts(gpu_meta.get('ts'))}")
        else:
            print("  GPU -> lower-bound fallback (no recent diagnostics)")
    else:
        print("- Diagnostics source: lower-bound fallback (no recent unschedulable diagnostics found)")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
