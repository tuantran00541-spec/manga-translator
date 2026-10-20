Build a durable job queue in this folder with the Python standard library only (no pip installs).

Parts:
1. queue_server.py: an HTTP server on port 8091 backed by SQLite (queue.db).
   - POST /jobs {"kind": "...", "payload": {...}, "max_attempts": N (default 5), "key": optional idempotency key} -> {"id": ...}.
     The same key twice returns the first job's id and creates nothing.
   - POST /lease {"worker": "name", "seconds": S} hands out the oldest ready job, at most one worker at a time per job.
     A lease that is not finished within S seconds expires and the job becomes ready again.
   - POST /jobs/<id>/done {"worker": ..., "result": ...} and POST /jobs/<id>/fail {"worker": ..., "error": ...}.
     Only the worker holding the current lease may finish or fail it (409 otherwise). A failed job is retried after
     exponential backoff with full jitter; after max_attempts it becomes "dead".
   - GET /jobs/<id> shows state (ready, leased, done, failed-waiting, dead), attempts, last error, result.
   - GET /metrics returns counts per state and the p50/p95 seconds from creation to done.
2. worker.py: a worker process that leases jobs and runs kinds "sleep" (payload {"seconds": x}), "fail_n"
   (fails its first n attempts, then succeeds) and "hash" (sha256 of payload["text"] repeated payload["times"] times).
3. Research the "full jitter" backoff formula from a reliable source on the web and cite it in README.md, with the
   formula as implemented.

Tests (pytest, in tests/): they start the server and workers as real subprocesses on free ports and prove, at least:
- 200 jobs with 4 workers all reach done exactly once (no job runs twice to completion, none is lost);
- a worker killed with SIGKILL in the middle of a job does not lose it: the job is leased again after expiry and done;
- idempotency keys, the 409 rule, retries with backoff (attempt times grow), and "dead" after max_attempts;
- the server restarted mid-run keeps every job and its state.

Done when `python3 -m pytest -q` passes and, when you finish, the server on port 8091 and two workers are left running,
with `curl http://127.0.0.1:8091/metrics` answering. Use whatever tools, skills or helper agents fit the work; the tests
and the research can go in parallel.
