# host-copy-bench

Measures aggregate host-to-GPU and GPU-to-host copy bandwidth when several GPUs copy **at the
same time**, and refuses to report a number when the copies did not actually overlap.

Written for the post
[Host-to-GPU bandwidth on an 8x RTX 4090 server without peer-to-peer](https://rockyshi.dev/posts/host-gpu-bandwidth-without-p2p/).

## Why the overlap check

The easy way to measure "four GPUs copying at once" is to start four processes, let each time
its own copies, and add up the four rates. If the processes do not start together, part of each
measurement is a GPU copying alone, and the sum can exceed what the hardware can carry. The
first version of the measurement behind the post got 37-41 GB/s through a link whose raw limit
is 31.5 GB/s in one direction: its processes had no common start, and each repetition copied to
the GPU and back, so the sum mixed directions and periods when a process copied alone.

Here every process waits on one barrier, copies for a fixed window, and records the window on the
host's monotonic clock. An aggregate is reported only if the window common to all processes
covers at least 99% of every process's own window; otherwise it is retried, up to twice, and
then that direction is reported as missing for the round.

## What it does

For each GPU set, one process per GPU:

1. Binds its CPU threads to the GPU's NUMA node and sets a memory policy so its pinned host
   buffers land on that node. The node comes from sysfs; if the GPU reports none, nothing is
   bound and placement is reported as "not checked".
2. Allocates two 256 MiB pinned host buffers and two device buffers, and asks the kernel
   (`move_pages`) which node every page of both buffers is on. Rounds with any page off the
   GPU's node are reported as "off-node" and left out of the medians.
3. Per direction, waits on a barrier and copies back to back for 3 s in 16 MiB chunks with two
   copies in flight per stream:
   - `h2d`: host to device only;
   - `bi`: host to device and device to host at once, on two streams.
4. Reports the sum over processes of bytes / own window, per direction, if the overlap check
   passes.

## Usage

Requirements: Linux on x86_64 or aarch64, NVIDIA GPUs, `nvidia-smi`, Python 3 with a CUDA
build of PyTorch.

```bash
# two GPUs: each alone, then both at once
python host_copy_bench.py --sets "0;1;0,1" --rounds 5 --json results.json

# the sets of the reference table below (8 GPUs, GPUs 0-3 on one socket, 4-7 on the other)
python host_copy_bench.py --sets "0;0,1;0,1,2,3;0,4;0,1,4,5;all" --rounds 5 --json results.json
```

`--sets` is a `;`-separated list of GPU sets, using `nvidia-smi` indices; `all` means every GPU.
Other options: `--seconds`, `--buffer-mib`, `--chunk-mib`, `--depth`, `--overlap`; see `--help`.
The JSON file holds every attempt's per-process windows and byte counts, the page placement,
the GPU models and the software versions, and is rewritten after every set. It also records GPU
UUIDs and the values of any inherited `CUDA_*`, `PYTORCH_*`, `OMP_*`, `NCCL_*` and `LD_PRELOAD`
variables, so check it before sharing it.

Output, one line per round and set, then medians:

```
round 1  GPUs 0,1: h2d 25.33  bi 17.91 + 18.63  placement ok
...
median over rounds, GB/s (bi = host-to-device + device-to-host); rounds with off-node buffer pages are excluded
  GPUs 0,1             h2d: 25.3  (n=5)   bi: 17.9 + 18.6  (n=5)
```

Run it on an otherwise idle machine: anything else using PCIe or host memory bandwidth shows up
in the result. The script refuses a GPU that already has compute processes unless you pass
`--allow-busy`, and warns about inherited `CUDA_*`, `PYTORCH_*`, `OMP_*`, `NCCL_*` and
`LD_PRELOAD` variables, which can change copy behaviour.

## Reference results

8x RTX 4090 (24 GB), two Xeon Gold 5320 sockets, no GPU peer-to-peer, four GPUs behind one PCIe
4.0 x16 switch uplink per socket. Output of the second command above on an idle machine
(2026-10-11), median of five rounds, GB/s:

| GPUs copying at once | h2d | bi (h2d + d2h) |
|---|---|---|
| 0 | 25.2 | 16.3 + 19.8 |
| 0,1 | 25.3 | 17.9 + 18.6 |
| 0,1,2,3 | 25.8 | 18.3 + 18.3 |
| 0,4 | 50.5 | 32.7 + 39.5 |
| 0,1,4,5 | 50.7 | 35.8 + 37.3 |
| all 8 | 51.6 | 36.7 + 36.6 |

The numbers in the post come from an earlier run of the same measurement on this machine; every
median agrees with this table to within 1.3%.

## License

MIT
