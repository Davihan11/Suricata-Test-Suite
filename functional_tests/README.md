# Functional Tests

This directory holds **functional** tests — tests that verify the *correctness* of
Suricata behaviour (e.g. RSS queueing, flow hashing) rather than throughput /
drop-rate performance.

The performance (throughput / drop-rate) tests live in `../performance_tests/`.

## Planned tests

- RSS queueing correctness:
  - same-flow packets land in the same worker queue
  - distinct flows spread across queues
  - bidirectional (symmetric RSS) flows hash to the same queue
