import json, numpy as np, pathlib, warnings
warnings.filterwarnings("ignore")
from bench.oof import oof_scores, text_pipeline
from bench.bootstrap import fold_assignments
from bench.run_mde import score_icc, calibrate, power_at, bootstrap_draws, CONDITIONS
FIXED = 0.58
mde = {(c["dataset"],c["model"]): c for c in
       json.loads(pathlib.Path("results/mde.json").read_text())["conditions"]}
print(f"  MDE recomputed against a FIXED floor top of {FIXED}\n")
print(f"  {'condition':22s} {'own floor':>10s} {'MDE own':>9s} {'MDE fixed':>10s} {'shift':>7s}", flush=True)
rows=[]
for ds, m, stem in CONDITIONS:
    c = mde.get((ds,m))
    if c is None or c["mde"] is None: continue
    t=json.loads(pathlib.Path(f"results/trials/{stem}.json").read_text())["trials"]
    X=[" ".join((x.get("readout") or "").split()) for x in t]
    y=np.array([bool(x["swayed"]) for x in t]); g=np.array([x["claim"] for x in t])
    s=oof_scores(X,y,g,text_pipeline()); fold=fold_assignments(y,g); icc=score_icc(s,g)
    _,inv=np.unique(g,return_inverse=True); nc=len(np.unique(g))
    draws=bootstrap_draws(y,g,fold,600,1)
    lo,hi=0.55,0.95
    for _ in range(5):
        mid=(lo+hi)/2
        d,_a=calibrate(y,inv,nc,icc,fold,mid,0)
        p,_u=power_at(y,g,fold,icc,d,FIXED,60,600,1,draws=draws)
        if p>=0.80: hi=mid
        else: lo=mid
    rows.append((f"{ds}/{m}", c["floor_hi"], c["mde"], hi))
    print(f"  {ds+'/'+m:22s} {c['floor_hi']:10.3f} {c['mde']:9.3f} {hi:10.3f} {hi-c['mde']:+7.3f}", flush=True)
o1=[r[0] for r in sorted(rows,key=lambda r:r[2])]; o2=[r[0] for r in sorted(rows,key=lambda r:r[3])]
print(f"\n  ordering, own floors : {o1}")
print(f"  ordering, fixed floor: {o2}")
print(f"  ORDERING PRESERVED: {o1==o2}")
json.dump({"fixed_floor":FIXED,"rows":[{"condition":r[0],"own_floor_hi":r[1],
          "mde_own":r[2],"mde_fixed":r[3]} for r in rows],"ordering_preserved":o1==o2},
          open("results/mde_fixed_floor.json","w"), indent=2)
