- PolarGrid qwen-3.8-27b, 2 full runs: 21/89 and 16/89; only 9 tasks passed in both. 13 and 8 AgentTimeoutError.
  30 tool calls were unreadable (the model wrote XML-like `<invoke name=...>` or `function_calls>` text instead of
  JSON); 8 times the server dropped tool calls while streaming; 4 tasks ended under 5K tokens with no error
  (the model stopped early); `create` was called 395 times (34 failed: bad names, syntax errors, truncated code);
  built-in skills were read only 5 times in 89 tasks; 35 times the harness sent "Out of time; write your final report".
- CommandCode xiaomi/mimo-v2.6-pro: 48/70 before the run was cancelled; 7 AgentTimeoutError; no rate limiting.
- reflection/beam:free failed every task until the harness learned to send back reasoning_content for every turn.
- reflection/beam:free at max reasoning, after the reasoning fix: 59/89 (2 timeouts, 27 rate-limit retries).
  28 of 30 failures ended with the model saying it was done. Already fixed since: jobs (servers the task wanted)
  were stopped when the session closed; polling a running job tripped the repeat guard; create refused loose names;
  bash/create_file were unknown; read_file on a folder was an error.
