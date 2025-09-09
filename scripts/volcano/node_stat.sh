python3 - <<'PY'
import json, subprocess, os, sys
from collections import Counter, defaultdict

def run(cmd):
    return subprocess.check_output(cmd, text=True)

def cpu_m(v):
    if not v: return 0
    s=str(v).strip()
    return int(s[:-1]) if s.endswith("m") else int(float(s)*1000)
def to_int(v):
    if not v: return 0
    s=str(v).strip()
    try: return int(s)
    except:
        try: return int(float(s))
        except: return 0

def is_ready(n):
    for c in n.get("status",{}).get("conditions",[]) or []:
        if c.get("type")=="Ready":
            return c.get("status")=="True"
    return False

def is_cordoned(n):
    return bool(n.get("spec",{}).get("unschedulable", False))

def is_control_plane(n):
    labels = n.get("metadata",{}).get("labels",{}) or {}
    return ("node-role.kubernetes.io/control-plane" in labels) or ("node-role.kubernetes.io/master" in labels)

def gpu_key_from(nodes):
    keys=set()
    for n in nodes:
        alloc=(n.get("status",{}).get("allocatable",{}) or {})
        for k in alloc.keys():
            if k.endswith("/gpu"): keys.add(k)
    for pref in ("nvidia.com/gpu","amd.com/gpu","intel.com/gpu"):
        if pref in keys: return pref
    return next(iter(keys), "nvidia.com/gpu")

def safe_get(d, *path, default=""):
    for p in path:
        d = (d or {}).get(p, {})
    return d or default

# --- get nodes ---
try:
    nodes = json.loads(run(["kubectl","get","nodes","-o","json"]))["items"]
except subprocess.CalledProcessError:
    print("Error: cannot 'kubectl get nodes -o json'", file=sys.stderr)
    sys.exit(1)

GPU_KEY = gpu_key_from(nodes)

# --- Aggregations (node-only) ---
by_kind_sched = Counter()                # (Kind, Schedulable)
by_status = Counter()                    # Ready/NotReady + Cordoned/Not
by_role = Counter()                      # label role groups
by_zone = Counter()                      # topology.kubernetes.io/zone
by_arch = Counter()                      # kubernetes.io/arch
by_kubelet = Counter()                   # kubeletVersion
by_instance = Counter()                  # node.kubernetes.io/instance-type
by_gpu_product = Counter()               # nvidia.com/gpu.product
taint_presence = Counter()               # has NoSchedule/NoExecute taints
features = Counter()                     # rdma, hugepages, gpu-present, etc.

def get_label(n, k):
    return (n.get("metadata",{}).get("labels",{}) or {}).get(k, "")

for n in nodes:
    name = n["metadata"]["name"]
    ready = is_ready(n)
    cordoned = is_cordoned(n)
    sched = "Schedulable" if (ready and not cordoned) else "Unschedulable"

    labels = n.get("metadata",{}).get("labels",{}) or {}
    alloc  = n.get("status",{}).get("allocatable",{}) or {}

    alloc_gpu = to_int(alloc.get(GPU_KEY,"0"))
    gpu_present = alloc_gpu > 0

    kind = "ControlPlane" if is_control_plane(n) else ("GPU" if gpu_present else "CPU")
    by_kind_sched[(kind, sched)] += 1

    # Status bucket: Ready/Cordoned matrix
    by_status[( "Ready" if ready else "NotReady", "Cordoned" if cordoned else "Open" )] += 1

    # Roles (common label keys)
    role = ( get_label(n, "kubernetes.io/role")
             or ("control-plane" if is_control_plane(n) else
                 ("gpu" if gpu_present else "worker")) )
    by_role[role] += 1

    # Zones, arch, kubelet, instance type
    by_zone[get_label(n, "topology.kubernetes.io/zone") or "(none)"] += 1
    by_arch[get_label(n, "kubernetes.io/arch") or "(unknown)"] += 1
    by_kubelet[n.get("status",{}).get("nodeInfo",{}).get("kubeletVersion","(unknown)")] += 1
    inst = ( get_label(n, "node.kubernetes.io/instance-type")
             or get_label(n, "beta.kubernetes.io/instance-type")
             or "(unknown)" )
    by_instance[inst] += 1

    # GPU product (if labeled)
    prod = get_label(n, "nvidia.com/gpu.product")
    if gpu_present:
        by_gpu_product[prod or "(unspecified)"] += 1

    # Features
    if gpu_present: features["gpu.present"] += 1
    # RDMA (common allocatable or labels vary by cluster)
    if to_int(alloc.get("rdma.hca","0")) > 0 or get_label(n,"rdma/hca"):
        features["rdma.present"] += 1
    # Hugepages
    if to_int(alloc.get("hugepages-1Gi","0"))>0 or to_int(alloc.get("hugepages-2Mi","0"))>0:
        features["hugepages.present"] += 1
    # Pod capacity (interesting density metric, no pods API needed)
    if to_int(alloc.get("pods","0"))>0:
        features["pod-capacity.reported"] += 1

# --- Optional: namespace-scoped "HasFree"/"Full" using requests (if allowed) ---
NS = os.environ.get("NS")
by_kind_sched_alloc = Counter()
if NS:
    try:
        pods = json.loads(run(["kubectl","get","pods","-n",NS,"-o","json"]))["items"]
        # sum requests per node
        req = defaultdict(lambda: {"cpu_m":0, "gpu":0})
        def pod_req(p):
            spec = p.get("spec",{}) or {}
            # sum over app containers
            cpu_m = sum(cpu_m(c.get("resources",{}).get("requests",{}).get("cpu","0"))
                        for c in spec.get("containers",[]) or [])
            gpu   = sum(to_int(c.get("resources",{}).get("requests",{}).get(GPU_KEY,"0"))
                        for c in spec.get("containers",[]) or [])
            # init containers: take element-wise max
            icpu = 0; igpu = 0
            for c in spec.get("initContainers",[]) or []:
                r = (c.get("resources",{}).get("requests",{}) or {})
                icpu = max(icpu, cpu_m(r.get("cpu","0")))
                igpu = max(igpu, to_int(r.get(GPU_KEY,"0")))
            return max(cpu_m, icpu), max(gpu, igpu)

        for p in pods:
            if p.get("status",{}).get("phase") in ("Failed","Succeeded"): continue
            node = p.get("spec",{}).get("nodeName")
            if not node: continue
            c,g = pod_req(p)
            req[node]["cpu_m"] += c
            req[node]["gpu"]   += g

        for n in nodes:
            name = n["metadata"]["name"]
            ready = is_ready(n); cord = is_cordoned(n)
            sched = "Schedulable" if (ready and not cord) else "Unschedulable"
            alloc = n.get("status",{}).get("allocatable",{}) or {}
            alloc_cpu = cpu_m(alloc.get("cpu","0"))
            alloc_gpu = to_int(alloc.get(GPU_KEY,"0"))
            gpu_present = alloc_gpu > 0
            kind = "ControlPlane" if is_control_plane(n) else ("GPU" if gpu_present else "CPU")
            if kind == "ControlPlane":
                by_kind_sched_alloc[(kind, sched, "n/a")] += 1
            else:
                r = req[name]
                has_free = (alloc_gpu > r["gpu"]) if gpu_present else (alloc_cpu > r["cpu_m"])
                by_kind_sched_alloc[(kind, sched, "HasFree" if has_free else "Full")] += 1
    except subprocess.CalledProcessError:
        pass  # user may not have pod access in NS; just skip the optional section

def print_section(title, items, headers=("KEY","COUNT"), keyfmt=lambda k:str(k)):
    print(f"== {title} ==")
    # compute column widths
    rows = [(keyfmt(k), str(v)) for k,v in items.items()]
    kW = max(len(headers[0]), *(len(r[0]) for r in rows)) if rows else len(headers[0])
    vW = max(len(headers[1]), *(len(r[1]) for r in rows)) if rows else len(headers[1])
    print(headers[0].ljust(kW), headers[1].rjust(vW))
    print("-"* (kW+1+vW))
    for k,v in sorted(items.items(), key=lambda kv:(str(kv[0]))):
        print(keyfmt(k).ljust(kW), str(v).rjust(vW))
    print()

print_section("Node kind × schedulability",
              by_kind_sched,
              headers=("KIND,SCHED","COUNT"),
              keyfmt=lambda k: f"{k[0]},{k[1]}")

print_section("Node status matrix (Ready/NotReady × Cordoned/Open)",
              by_status,
              headers=("STATUS","COUNT"),
              keyfmt=lambda k: f"{k[0]},{k[1]}")

print_section("Roles (label-based fallback)",
              by_role)

print_section("Zones (topology.kubernetes.io/zone)",
              by_zone)

print_section("Architectures (kubernetes.io/arch)",
              by_arch)

print_section("Kubelet versions",
              by_kubelet)

print_section("Instance types",
              by_instance)

print_section("GPU products (nvidia.com/gpu.product)",
              by_gpu_product)

print_section("Feature presence (gpu/rdma/hugepages/pod-capacity)",
              features)

if by_kind_sched_alloc:
    print_section(f"Namespace-scoped allocation in NS='{NS}' (HasFree/Full; ControlPlane=n/a)",
                  by_kind_sched_alloc,
                  headers=("KIND,SCHED,ALLOC","COUNT"),
                  keyfmt=lambda k: f"{k[0]},{k[1]},{k[2]}")
PY
