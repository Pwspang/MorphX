#!/usr/bin/env python3
"""Where the decode thread stalls in CUDA calls (nsys SQLite): time in calls >20 ms by name, and the
prefill thread call overlapping each long cudaEventCreate. Usage: nsys_host_stalls.py <trace.sqlite>"""
import sqlite3, collections, sys
db=sqlite3.connect(sys.argv[1])
S=dict(db.execute("select id,value from StringIds"))
k=db.execute("select k.streamId, r.globalTid, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=k.correlationId group by 1,2").fetchall()
k.sort(key=lambda x:-x[2]); tids=[t for _,t,_ in k[:2]]
calls=collections.defaultdict(list)
for s,e,t,n in db.execute("select start,end,globalTid,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    calls[t].append((s,e,S[n].split('_v')[0]))
# decode thread = the one with more kernel launches
dec,pre=tids[0],tids[1]
d,p=sorted(calls[dec]),sorted(calls[pre])
tot=collections.Counter()
for s,e,n in d:
    if e-s>20e6: tot[n]+=(e-s)/1e9
print(sys.argv[1], "decode-thread time in CUDA calls >20 ms (s):", {k:round(v,1) for k,v in tot.items()})
ev=[(s,e) for s,e,n in d if n=="cudaEventCreateWithFlags" and e-s>20e6]
c=collections.Counter()
for a,b in ev:
    for s,e,n in p:
        if e>a and s<b and min(b,e)-max(a,s)>5e6: c[n]+=1
print("  prefill-thread call during each long cudaEventCreate:", c.most_common(3), "of", len(ev))
