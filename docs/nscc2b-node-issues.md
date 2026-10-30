# nscc2b (ASPIRE2B) node issues seen by the MorphX port

Log of PBS jobs that hung or misbehaved on nscc2b compute nodes, with enough detail to report
to NSCC support. Project 12004493 (paid GPU hours), user wpang013, queue `normal` (routed to
`gdev` / `g1`). Times are SGT.

## Symptom (start-up hang)

The same pattern every time:
1. PBS shows the job as running (`R`) with a node assigned (`exec_host`), but
   `resources_used.cput` stays at 0. In some cases `resources_used.walltime` freezes too
   (job 195376: stuck at 00:02:12). One report notes 716 kB of memory used.
2. The job script never starts. Its first commands (`mkdir` of the output directory on
   `/home/users/ntu/wpang013/scratch`, then a log line) never happen, so no output appears.
3. `qdel` often times out or hangs. The job then lingers in the exiting state (`E`), still
   holding its GPU, for minutes or longer. Job 191936 held a GPU for about 16 minutes.
4. `pbsnodes` still reports the node as `state = free`, and PBS keeps scheduling jobs onto it.

The same scripts and inputs run normally on other nodes, and the hang happens before any of our
code runs. It is a node-side problem in PBS job start-up: the MOM, the prologue, GPU or cgroup
setup, or the shared filesystem. We can't inspect compute nodes: an ssh login to a node, even
a healthy one, returned no command output. Only NSCC staff can diagnose it from the MOM logs
and kernel messages.

It is **intermittent per node**. Several of these nodes ran other jobs of ours normally:
- gpu064: 190793, 192350
- gpu080: 192663
- gpu089: 186837, 192550
- gpu096: 192549 ran; 192658 was slow to start but finished

## Incidents

| Date | Job | Node | What happened |
|---|---|---|---|
| 2026-10-27 | 191936 | asp2b-gpu089 | 0 CPU; `qdel` hung; held a billed GPU for ~16 min before reaching exit |
| 2026-10-27 | 192050 | asp2b-gpu089 | 0 CPU; deleted after ~2 min (15 s recorded) |
| 2026-10-27 | 192231 | asp2b-gpu098 | 0 CPU, no log for 4 min; deleted, resubmitted as 192255 |
| 2026-10-27 | 192282 | asp2b-gpu096 | 0 CPU, 716 kB memory, no log after 3–6 min; deleted |
| 2026-10-27 | 192302 | asp2b-gpu155 | same as 192282; deleted |
| 2026-10-27 | 192334 | asp2b-gpu155 | same; after `qdel` still stuck in `E` at 20:04 |
| 2026-10-27 | 192313 | asp2b-gpu080 | 0 CPU; `qdel` timed out twice before PBS cleared it |
| 2026-10-27 | 192548 | asp2b-gpu064 | no CPU time, no output; deleted, resubmitted as 192555 |
| 2026-10-27 | 192658 | asp2b-gpu096 | no log by 22:30 and `qdel` timed out; the job later started and finished normally |
| 2026-10-27 | 192350 | asp2b-gpu064 | ran and finished, but stayed in `E` afterwards; `qdel` timed out |
| 2026-10-28 15:48 | 195376 | asp2b-gpu109 | 0 CPU, walltime frozen at 00:02:12, no output; after `qdel` stuck in `E` (Exit_status 143), still listed in gpu109's `jobs` with its GPU assigned |
| 2026-10-28 ~21:00 | 196337 | asp2b-gpu081 | `R` but no `resources_used` at all after 7+ min, no output; `qdel` timed out (30 s); resubmitted as 196372 |
| 2026-10-29 11:09 | 200642 | asp2b-gpu072 | went `R` → `E` within a minute with `Exit_status = -3` (start-up failed, PBS retries); no output; PBS requeued it (`run_count = 2`) |
| 2026-10-29 11:10 | 200642 (rerun) | asp2b-gpu046 | `R` for 9 min with cput, mem and walltime all 0, no output; `qdel` (11:19) timed out (40 s); resubmitted as 200734. The script then started at 11:24 anyway and ran 10 min, until the 11:19 `qdel` took effect at 11:34 and killed it mid-run (Exit_status 0). Rerun as 200788 |
| 2026-10-29 11:25 | 200734 | asp2b-gpu046 | `R` with no resources reported; `qdel` and `qdel -W force` both timed out (40 s / 60 s) |
| 2026-10-29 11:26 | 200769 | asp2b-gpu046 | `R` with no resources reported after 14 min, no output; `qdel` timed out (20 s); resubmitted as 200832 |
| 2026-10-29 11:26 | 200772 | asp2b-gpu138 | `R`, cput 0 at 3.5 min walltime, no output; `qdel` timed out (20 s); resubmitted as 200833 |
| 2026-10-29 11:26 | 200773 | asp2b-gpu058 → gpu091 | first start on gpu058 failed and PBS requeued it (`run_count = 2`); on gpu091 cput 0 at 7 min walltime, no output; `qdel` timed out (20 s); resubmitted as 200834 |

## Different symptom: erratic results

| Date | Job | Node | What happened |
|---|---|---|---|
| 2026-10-27 | 192557 | asp2b-gpu131 | Ran, but latencies were erratic and inconsistent with two reruns on other nodes (192658, 192663). Example: stream 0.172 vs 0.022 s/token at the same point. Results discarded. |
| 2026-10-29 | 200788 | asp2b-gpu036 | Ran normally, but decode was ~25% slower at light load than the same points on gpu046 (job 200642): TBT 11.9 vs 9.3 ms, with the same TTFT. Looks like slower host-side work, not the GPU. Points kept but flagged. |

## How we handle it

- Every job writes a live log (`tee`) and is checked 1–5 minutes after it starts. A job at
  0 CPU with no log is deleted and resubmitted.
- Jobs are never pinned to a host. Pinning to avoid bad nodes once left a job queued for ~2 h.
- Each step inside a job runs under its own `timeout`, so a hang after start-up can't burn
  the whole walltime.

## Suggested report to NSCC support

Ask them to check the PBS MOM logs and kernel messages on gpu046, 058, 064, 072, 080, 081, 089, 091, 096, 098, 109, 138 and
155 around the listed times, and why `qdel`'d jobs stay in `E` while still holding GPUs. Also
ask whether the nodes should be drained, or a health check added so PBS stops scheduling onto
them.
