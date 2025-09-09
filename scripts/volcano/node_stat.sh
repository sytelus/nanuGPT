python3 - <<'PY'
import json, subprocess, os
from collections import Counter

def run(cmd): return subprocess.check_output(cmd, text=True)
def cpu_m(v):
    if not v: return 0
    s=str(v).strip()
    return int(s[:-1]) if s.endswith("m") else int(float(s)*1000)
def i(v):
    if not v or str(v).strip()=="": return 0
    try: return int(str(v).strip())
    except: return 0

def get_ready(node):
    for c in node.get("status",{}).get("conditions",[]):
        if c.get("type")=="Ready":
            return c.get("status")=="True"
    return False
def is_cordoned(node):
    return bool(node.get("spec",{}).get("unschedulable",False))

nodes=json.loads(run(["kubectl","get","nodes","-o","json"]))["items"]

# detect GPU key
keys=set()
for n in nodes: keys.update((n.get("status",{}).get("allocatable",{}) or {}).keys())
gpu_key="nvidia.com/gpu" if "nvidia.com/gpu" in keys else next((k for k in keys if k.endswith("/gpu")), "nvidia.com/gpu")

# classify and count
counts=Counter()
for n in nodes:
    name=n["metadata"]["name"]
    alloc=n.get("status",{}).get("allocatable",{}) or {}
    alloc_cpu=cpu_m(alloc.get("cpu","0")); alloc_gpu=i(alloc.get(gpu_key,"0"))
    ready=get_ready(n); cord=is_cordoned(n)
    sched="Schedulable" if (ready and not cord) else "Unschedulable"

    labels=n.get("metadata",{}).get("labels",{}) or {}
    if "node-role.kubernetes.io/control-plane" in labels or "node-role.kubernetes.io/master" in labels:
        kind="ControlPlane"
    elif alloc_gpu>0:
        kind="GPU"
    else:
        kind="CPU"

    counts[(kind,sched)]+=1

# pretty print
rows=[("KIND","SCHED","COUNT")]
for k in sorted(counts,key=lambda x:(x[0],x[1])):
    rows.append((k[0],k[1],str(counts[k])))
w=[max(len(r[i]) for r in rows) for i in range(3)]
for r in rows:
    print("  ".join(s.ljust(w[i]) for i,s in enumerate(r)))
PY
