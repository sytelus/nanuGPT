python3 - <<'PY'
import json, subprocess, os, sys
from collections import Counter, defaultdict

def run(cmd): return subprocess.check_output(cmd, text=True)

def cpu_m(v):
    if not v: return 0
    s=str(v).strip()
    return int(s[:-1]) if s.endswith("m") else int(float(s)*1000)

def to_int(v):
    if v is None: return 0
    s=str(v).strip()
    if s=="": return 0
    try: return int(s)
    except:
        try: return int(float(s))
        except: return 0

def is_ready(n):
    for c in n.get("status",{}).get("conditions",[]) or []:
        if c.get("type")=="Ready":
            return c.get("status")=="True"
    return False

def is_cordoned(n): return bool(n.get("spec",{}).get("unschedulable",False))

def is_cp(n):
    L=n.get("metadata",{}).get("labels",{}) or {}
    return "node-role.kubernetes.io/control-plane" in L or "node-role.kubernetes.io/master" in L

def detect_gpu_key(nodes):
    keys=set()
    for n in nodes:
        for k in (n.get("status",{}).get("allocatable",{}) or {}).keys():
            if k.endswith("/gpu"): keys.add(k)
    for pref in ("nvidia.com/gpu","amd.com/gpu","intel.com/gpu"):
        if pref in keys: return pref
    return next(iter(keys), "nvidia.com/gpu")

def pod_req(p, gpu_key):
    spec=p.get("spec",{}) or {}
    # sum app containers
    c_cpu=sum(cpu_m((c.get("resources",{}).get("requests",{}) or {}).get("cpu","0")) for c in spec.get("containers",[]) or [])
    c_gpu=sum(to_int((c.get("resources",{}).get("requests",{}) or {}).get(gpu_key,"0")) for c in spec.get("containers",[]) or [])
    # init containers -> element-wise max
    i_cpu=i_gpu=0
    for c in spec.get("initContainers",[]) or []:
        r=(c.get("resources",{}).get("requests",{}) or {})
        i_cpu=max(i_cpu, cpu_m(r.get("cpu","0")))
        i_gpu=max(i_gpu, to_int(r.get(gpu_key,"0")))
    return max(c_cpu,i_cpu), max(c_gpu,i_gpu)

# --- nodes ---
try:
    nodes=json.loads(run(["kubectl","get","nodes","-o","json"]))["items"]
except subprocess.CalledProcessError:
    print("Error: cannot 'kubectl get nodes -o json'", file=sys.stderr); sys.exit(1)

GPU_KEY = detect_gpu_key(nodes)

# --- pods (namespace-scoped via VOLCANO_NAMESPACE, else cluster-wide if permitted) ---
ns=os.environ.get("VOLCANO_NAMESPACE","")
pods=[]
try:
    if ns:
        pods=json.loads(run(["kubectl","get","pods","-n",ns,"-o","json"]))["items"]
    else:
        pods=json.loads(run(["kubectl","get","pods","-A","-o","json"]))["items"]
except subprocess.CalledProcessError:
    # no pod access -> we can still print schedulability, but availability will be Unknown
    pods=[]

# sum requests per node
req=defaultdict(lambda: {"cpu_m":0,"gpu":0})
for p in pods:
    if p.get("status",{}).get("phase") in ("Succeeded","Failed"): continue
    node=(p.get("spec",{}) or {}).get("nodeName")
    if not node: continue
    c,g=pod_req(p,GPU_KEY)
    req[node]["cpu_m"]+=c
    req[node]["gpu"]+=g

# ---- summaries ----
sched_summary = Counter()               # (Kind, Schedulable/Unschedulable) -> node count
avail_summary = Counter()               # (Kind, Available/Full/Unknown) -> node count (schedulable workers only)
gpu_alloc_total = 0                     # sum of allocatable GPUs on schedulable GPU nodes
gpu_free_total  = 0                     # sum of (alloc - requested) on schedulable GPU nodes

for n in nodes:
    name=n["metadata"]["name"]
    ready=is_ready(n); cord=is_cordoned(n)
    sched = "Schedulable" if (ready and not cord) else "Unschedulable"

    alloc = n.get("status",{}).get("allocatable",{}) or {}
    alloc_cpu=cpu_m(alloc.get("cpu","0"))
    alloc_gpu=to_int(alloc.get(GPU_KEY,"0"))
    kind = "ControlPlane" if is_cp(n) else ("GPU" if alloc_gpu>0 else "CPU")

    # schedulability summary (includes Unschedulable)
    sched_summary[(kind, sched)] += 1

    # availability summary only for *schedulable* worker nodes
    if sched != "Schedulable":
        continue
    if kind == "ControlPlane":
        continue

    if not pods:  # no visibility into pod requests
        avail_summary[(kind, "Unknown")] += 1
    else:
        r=req[name]
        if kind=="GPU":
            free = max(0, alloc_gpu - r["gpu"])
            gpu_alloc_total += alloc_gpu
            gpu_free_total  += free
            avail_summary[(kind, "Available" if free>0 else "Full")] += 1
        else:  # CPU
            free = max(0, alloc_cpu - r["cpu_m"])
            avail_summary[(kind, "Available" if free>0 else "Full")] += 1

# ---- print helpers ----
def print_table(title, rows, headers):
    cols=len(headers)
    widths=[len(h) for h in headers]
    for r in rows:
        for i,s in enumerate(r):
            widths[i]=max(widths[i], len(str(s)))
    print(title)
    print("  "+"  ".join(h.ljust(widths[i]) for i,h in enumerate(headers)))
    print("  "+"  ".join("-"*widths[i] for i in range(cols)))
    for r in rows:
        print("  "+"  ".join(str(s).ljust(widths[i]) for i,s in enumerate(r)))
    print()

# Schedulability table (node counts)
sched_rows=[(k[0], k[1], sched_summary[k]) for k in sorted(sched_summary, key=lambda x:(x[0], x[1]))]
print_table("Schedulability (node counts)", sched_rows, ["KIND","SCHED","COUNT"])

# Availability table (schedulable workers; node counts)
avail_rows=[(k[0], k[1], avail_summary[k]) for k in sorted(avail_summary, key=lambda x:(x[0], x[1]))]
if avail_rows:
    print_table("Availability among Schedulable worker nodes (node counts)", avail_rows, ["KIND","STATE","COUNT"])
else:
    print("Availability among Schedulable worker nodes (node counts)\n  (no data or no schedulable workers)\n")

# GPU capacity line
if gpu_alloc_total or gpu_free_total:
    print(f"GPU capacity (schedulable GPU nodes): total_allocatable={gpu_alloc_total}, total_free_by_requests={gpu_free_total}")
PY
