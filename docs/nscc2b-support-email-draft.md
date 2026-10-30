Subject: ASPIRE2B (project 12004493): GPU jobs hang at start-up on some nodes; deleted jobs stay in E state

Hello NSCC support,

Since 27 October, some of our GPU jobs on ASPIRE2B (user wpang013, project 12004493) have hung at
start-up on particular compute nodes. The same scripts and inputs run normally on other nodes.

**What we see**
1. The job is shown as running (`R`) with a node assigned, but `resources_used.cput` stays at 0.
   On job 195376, `resources_used.walltime` also stopped advancing (stuck at 00:02:12).
2. The job script never starts. Its first command, creating an output directory under
   `/home/users/ntu/wpang013/scratch`, never runs, and no output is written.
3. `qdel` often hangs or times out. The job then stays in the `E` state while still holding its GPU.
   Job 191936 held a GPU for about 16 minutes this way. Job 195376 is still listed in the `jobs` of
   asp2b-gpu109 with a GPU assigned.
4. `pbsnodes` reports the node as `state = free`, and new jobs continue to be scheduled on it.

**Affected jobs (times SGT)**

| Date | Job | Node | Observation |
|---|---|---|---|
| 27 Oct | 191936 | asp2b-gpu089 | 0 CPU; `qdel` hung; held the GPU for ~16 min |
| 27 Oct | 192050 | asp2b-gpu089 | 0 CPU |
| 27 Oct | 192231 | asp2b-gpu098 | 0 CPU, no output after 4 min |
| 27 Oct | 192282 | asp2b-gpu096 | 0 CPU, 716 kB memory, no output after 3–6 min |
| 27 Oct | 192302 | asp2b-gpu155 | same |
| 27 Oct | 192334 | asp2b-gpu155 | same; after `qdel` still in `E` at 20:04 |
| 27 Oct | 192313 | asp2b-gpu080 | 0 CPU; `qdel` timed out twice |
| 27 Oct | 192548 | asp2b-gpu064 | 0 CPU, no output |
| 27 Oct | 192350 | asp2b-gpu064 | finished normally but stayed in `E`; `qdel` timed out |
| 28 Oct 15:48 | 195376 | asp2b-gpu109 | 0 CPU, walltime frozen at 00:02:12, no output; in `E` after `qdel` (Exit_status 143) |
| 28 Oct ~21:00 | 196337 | asp2b-gpu081 | `R` with no resources_used reported, no output; `qdel` timed out |

The problem is intermittent: several of these nodes (gpu064, 080, 089, 096) also ran other jobs of
ours normally.

**Requests**
1. Could you check the PBS MOM logs and kernel messages on these nodes around the times above?
2. Could you clear any of our jobs still stuck in `E` and release their GPUs?
3. If a node is unhealthy, could it be drained, or a health check added so jobs are no longer
   scheduled on it?
4. Since GPU time on this project is paid, could the time charged to the jobs above be reviewed?

Thank you,
Pang Wei Siong (wpang013)
