# Unified NVIDIA GPU Telemetry Metrics Cheat Sheet

## Purpose
When monitoring NVIDIA GPUs, it's common to combine high‑granularity counters from DCGM (`dcgmi`) with the more ubiquitous `nvidia‑smi`. To simplify post‑processing, the tables below align the two toolchains so that each column/field position carries the same semantic meaning.

## Core metrics – aligned order

| Pos | Metric description | `nvidia‑smi --query-gpu` property | DCGM field ID | DCGM tag |
|----|----------------|-------------------------------|--------------|----------------------------|
| 0 | Host timestamp | `timestamp` | — | (added automatically by `dcgmi dmon`) |
| 1 | GPU index | `index` | 52 | `DCGM_FI_DEV_NVML_INDEX` |
| 2 | GPU name | `name` | 50 | `DCGM_FI_DEV_NAME` |
| 3 | Board power draw (W) | `power.draw` | 155 | `DCGM_FI_DEV_POWER_USAGE` |
| 4 | Power limit (W) | `power.limit` | 160 | `DCGM_FI_DEV_POWER_MGMT_LIMIT` |
| 5 | Core temperature (°C) | `temperature.gpu` | 150 | `DCGM_FI_DEV_GPU_TEMP` |
| 6 | GPU utilisation (%) | `utilization.gpu` | 203 | `DCGM_FI_DEV_GPU_UTIL` |
| 7 | Memory‑interface busy (%) | `utilization.memory` | 204 | `DCGM_FI_DEV_MEM_COPY_UTIL` |
| 8 | Frame‑buffer memory total (MiB) | `memory.total` | 250 | `DCGM_FI_DEV_FB_TOTAL` |
| 9 | Frame‑buffer memory free (MiB) | `memory.free` | 251 | `DCGM_FI_DEV_FB_FREE` |
| 10 | Frame‑buffer memory used (MiB) | `memory.used` | 252 | `DCGM_FI_DEV_FB_USED` |
| 11 | SM clock (MHz) | `clocks.sm` | 100 | `DCGM_FI_DEV_SM_CLOCK` |
| 12 | Memory clock (MHz) | `clocks.mem` | 101 | `DCGM_FI_DEV_MEM_CLOCK` |
| 13 | Graphics clock (MHz) | `clocks.gr` | 100* | (SM clock used as proxy) |
| 14 | Application graphics clock (MHz) | `clocks.applications.graphics` | 110 | `DCGM_FI_DEV_APP_SM_CLOCK` |
| 15 | Application memory clock (MHz) | `clocks.applications.memory` | 111 | `DCGM_FI_DEV_APP_MEM_CLOCK` |
| 16 | Performance state | `pstate` | 190 | `DCGM_FI_DEV_PSTATE` |

\* Data‑centre GPUs report identical SM and graphics clocks; use field 100 as a convenient surrogate. If you need the dedicated graphics clock value, query NVML directly.

### Python list definitions

```python
# nvidia‑smi properties
SMI_QUERY = [
    "timestamp", "index", "name",
    "power.draw", "power.limit",
    "temperature.gpu",
    "utilization.gpu", "utilization.memory",
    "memory.total", "memory.free", "memory.used",
    "clocks.sm", "clocks.mem", "clocks.gr",
    "clocks.applications.graphics", "clocks.applications.memory",
    "pstate"
]

# DCGM field IDs (aligned to the same indices)
DCGM_FIELDS = [
    "timestamp",      # inserted automatically by dcgmi
    52, 50,
    155, 160,
    150,
    203, 204,
    250, 251, 252,
    100, 101, 100,
    110, 111,
    190
]
```

## Extended DCGM‑only counters

Add these after the core list to capture fine‑grained activity and cumulative stats:

```python
EXTRA_DCGM = [
    140,            # memory (HBM) temperature
    156,            # total energy consumption (mJ)
    1001, 1002,     # graphics & SM active %
    1003, 1004,     # SM occupancy, tensor pipe active
    1005, 1006,     # DRAM active, FP64 active
    1007, 1008,     # FP32 active, FP16 active
    1009, 1010      # PCIe TX / RX bytes
]
```

## Collection examples

```bash
# Collect aligned metrics at 1 Hz to CSV
dcgmi dmon -d 1 -e $(IFS=,; echo "${DCGM_FIELDS[*]}") -c 0 -f dcgm_core.csv

nvidia-smi --query-gpu=$(IFS=,; echo "${SMI_QUERY[*]}")            --format=csv,noheader -l 1 > smi_core.csv

# Optional: gather extended DCGM counters in parallel
dcgmi dmon -d 1 -e $(IFS=,; echo "${EXTRA_DCGM[*]}") -c 0 -f dcgm_extra.csv
```

### Tips & caveats

* **Timestamps** — Both tools log host time, so aligning by wall‑clock is straightforward.
* **Granularity** — DCGM’s profiler fields (100x) offer cycle‑accurate activity; prefer them over coarse 203/204 where supported.
* **Memory utilisation** — DCGM’s 204 reflects memcpy engine utilisation; 1005 gives overall DRAM busy time.
* **MIG environments** — Replace 203/204 with 1001/1005 when monitoring MIG instances; the legacy utilisation counters are not MIG‑aware.
* **Power averaging** — On Ampere and newer data‑centre GPUs, `power.draw` is already a 1 second rolling mean; older GPUs expose an instantaneous sample.

## References
* *NVIDIA System Management Interface (nvidia‑smi) User Guide*, v555.xx
* *NVIDIA Data‑Centre GPU Manager (DCGM) API Reference*, v3.x

---

Copyright © 2025 Mert. Licensed under CC‑BY‑4.0.
