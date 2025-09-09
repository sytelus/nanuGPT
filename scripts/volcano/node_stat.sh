python3 - <<'PY'
import json, subprocess, os
def run(cmd): return subprocess.check_output(cmd, text=True)
def cpu_m(v):
    if v is None: return 0
    s=str(v).strip()
    return int(s[:-1]) if s.endswith('m') else int(float(s)*1000)
def i(v):
    if v is None or str(v).strip()=="": return 0
    try: return int(str(v).strip())
    except: return 0

nodes = json.loads(run(["kubectl","get","nodes","-o","json"]))["items"]
ns = os.getenv("NS", "")
pods = json.loads(run(["kubectl","get","pods","-n",ns,"-o","json"]))["items"]

# detect GPU resource key
keys=set()
for n in nodes:
    keys.update((n.get("status",{}).get("allocatable",{}) or {}).keys())
gpu_key = "nvidia.com/gpu" if "nvidia.com/gpu" in keys else next((k for k in keys if k.endswith("/gpu")), "nvidia.com/gpu")

# sum requests per node for this namespace only
req={}
for p in pods:
    n=p.get("spec",{}).get("nodeName");  ph=p.get("status",{}).get("phase")
    if not n or ph in ("Succeeded","Failed"): continue
    cpu=sum(cpu_m((c.get("resources",{}).get("requests",{}) or {}).get("cpu","0")) for c in p["spec"].get("containers",[]))
    gpu=sum(i((c.get("resources",{}).get("requests",{}) or {}).get(gpu_key,"0")) for c in p["spec"].get("containers",[]))
    req.setdefault(n,{"cpu":0,"gpu":0})
    req[n]["cpu"]+=cpu; req[n]["gpu"]+=gpu

from collections import Counter
cnt=Counter()
for n in nodes:
    name=n["metadata"]["name"]
    alloc=n.get("status",{}).get("allocatable",{}) or {}
    alloc_cpu=cpu_m(alloc.get("cpu","0")); alloc_gpu=i(alloc.get(gpu_key,"0"))
    ready = any(c.get("type")=="Ready" and c.get("status")=="True" for c in n.get("status",{}).get("conditions",[]))
    cordoned = bool(n.get("spec",{}).get("unschedulable",False))
    sched = "Schedulable" if (ready and not cordoned) else "Unschedulable"
    is_gpu = alloc_gpu>0
    r=req.get(name,{"cpu":0,"gpu":0})
    alloc_state = ("HasFree" if (alloc_gpu>r["gpu"] if is_gpu else alloc_cpu>r["cpu"]) else "Full")
    kind="GPU" if is_gpu else "CPU"
    cnt[(kind,sched,alloc_state)]+=1

rows=[("KIND","SCHED","ALLOC","COUNT")]
for k in sorted(cnt,key=lambda x:(x[0],x[1],x[2])): rows.append((k[0],k[1],k[2],str(cnt[k])))
w=[max(len(r[i]) for r in rows) for i in range(4)]
for r in rows: print("  ".join(s.ljust(w[i]) for i,s in enumerate(r)))
PY
