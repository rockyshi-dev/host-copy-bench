#!/usr/bin/env python3
"""Aggregate host<->GPU copy bandwidth when several GPUs copy at the same time.

One process per GPU. Each process binds its CPU threads to the GPU's NUMA node and allocates its
pinned host buffers there (set_mempolicy before allocation), then asks the kernel which node
every page of both buffers is on (move_pages). All processes wait on one barrier, then copy back
to back for a fixed window, either host to device only ("h2d") or both directions at once on two
streams ("bi"). Each process records its window on the host monotonic clock and counts the bytes
it finished.

An aggregate is the sum over processes of bytes / own window, per direction, and it is reported
only if the window shared by all processes covers at least --overlap of every process's window.
Without that check, per-process rates that were not measured at the same time get added up,
which can produce a "bandwidth" no link can carry. A failed check is retried up to twice; after
that the direction is reported as missing for that round.

Usage (8 GPUs, two sockets of four):
    python host_copy_bench.py --sets "0;0,1;0,1,2,3;0,4;0,1,4,5;all" --rounds 5 --json out.json
"""
import argparse
import collections
import ctypes
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time

MPOL_BIND = 2
# (set_mempolicy, move_pages) syscall numbers
SYSCALLS = {'x86_64': (238, 279), 'aarch64': (237, 239)}
# Inherited variables that can change copy behaviour; reported, and recorded in the JSON.
WATCHED_ENV = ('CUDA_', 'PYTORCH_', 'OMP_', 'NCCL_', 'LD_PRELOAD')
OWN_ENV = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')


# ---------------------------------------------------------------- topology (launcher side)

def nvidia_smi(fields):
    out = subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader'],
                         check=True, capture_output=True, text=True).stdout
    return [[s.strip() for s in line.split(',')] for line in out.strip().splitlines()]


def gpus():
    """{index: {bus_id (sysfs form), uuid, name}}, indices in nvidia-smi (PCI bus) order."""
    info = {}
    for index, bus, uuid, name in nvidia_smi('index,pci.bus_id,uuid,name'):
        domain, rest = bus.split(':', 1)  # nvidia-smi prints an 8-hex-digit domain, sysfs 4
        info[int(index)] = {'bus_id': f'{int(domain, 16):04x}:{rest}'.lower(), 'uuid': uuid, 'name': name}
    return info


def numa_nodes():
    return sorted(int(p.name[4:]) for p in Path('/sys/devices/system/node').glob('node[0-9]*'))


def numa_of(bus_id):
    """(NUMA node, CPU list) of a PCI device; (None, None) when the device reports none (-1)."""
    node = int((Path('/sys/bus/pci/devices') / bus_id / 'numa_node').read_text())
    if node < 0:
        return None, None
    cpus = (Path('/sys/devices/system/node') / f'node{node}' / 'cpulist').read_text().strip()
    return node, cpus


def parse_cpulist(text):
    cpus = set()
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def busy_gpus(uuids):
    """Compute processes already running on any of these GPUs."""
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout
    return [line for line in out.strip().splitlines() if line.split(',')[0].strip() in uuids]


# ---------------------------------------------------------------- worker (one per GPU)

def libc():
    return ctypes.CDLL(None, use_errno=True)


def bind_memory(node):
    nr = SYSCALLS[platform.machine()][0]
    bits = 8 * ctypes.sizeof(ctypes.c_ulong)
    mask = (ctypes.c_ulong * (node // bits + 1))()
    mask[node // bits] = 1 << (node % bits)
    # maxnode counts bits the kernel reads; it reads maxnode - 1 of them.
    if libc().syscall(ctypes.c_long(nr), ctypes.c_long(MPOL_BIND), mask, ctypes.c_ulong(node + 2)) != 0:
        raise OSError(ctypes.get_errno(), f'set_mempolicy(node {node}) failed')


def pages_per_node(tensor):
    """{node: pages} for every page of a host tensor, from move_pages(2) with nodes=NULL."""
    nr = SYSCALLS[platform.machine()][1]
    page = os.sysconf('SC_PAGE_SIZE')
    first = tensor.data_ptr() // page * page
    count = (tensor.data_ptr() + tensor.numel() * tensor.element_size() - first + page - 1) // page
    addrs = (ctypes.c_void_p * count)(*[first + i * page for i in range(count)])
    status = (ctypes.c_int * count)()
    if libc().syscall(ctypes.c_long(nr), ctypes.c_long(0), ctypes.c_ulong(count), addrs, None, status,
                      ctypes.c_long(0)) != 0:
        raise OSError(ctypes.get_errno(), 'move_pages failed')
    return dict(collections.Counter(int(s) for s in status))  # negative keys are -errno


def window(torch, direction, buffers, streams, seconds, chunk, depth):
    host_src, host_dst, dev_a, dev_b = buffers
    lanes = [('h2d', streams[0])] if direction == 'h2d' else [('h2d', streams[0]), ('d2h', streams[1])]
    slices = host_src.numel() // chunk
    queued = {d: collections.deque() for d, _ in lanes}
    copies = {d: 0 for d, _ in lanes}
    start = time.monotonic_ns()
    deadline = start + int(seconds * 1e9)
    while time.monotonic_ns() < deadline:
        for d, stream in lanes:
            q = queued[d]
            while q and q[0].query():
                q.popleft()
            if len(q) < depth:
                i = copies[d] % slices
                part = slice(i * chunk, (i + 1) * chunk)
                with torch.cuda.stream(stream):
                    if d == 'h2d':
                        dev_a[part].copy_(host_src[part], non_blocking=True)
                    else:
                        host_dst[part].copy_(dev_b[part], non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
                q.append(event)
                copies[d] += 1
    for _, stream in lanes:
        stream.synchronize()
    end = time.monotonic_ns()
    return {'start_ns': start, 'end_ns': end, 'copies': copies,
            'bytes': {d: n * chunk for d, n in copies.items()}}


def overlap(windows, minimum):
    starts = [w['start_ns'] for w in windows]
    ends = [w['end_ns'] for w in windows]
    if any(e <= s for s, e in zip(starts, ends)):
        raise RuntimeError('non-positive copy window')
    common = min(ends) - max(starts)
    coverage = [max(0, common) / (e - s) for s, e in zip(starts, ends)]
    return all(c >= minimum for c in coverage), coverage


def aggregates(windows):
    """Bytes per second per direction: the sum over processes of bytes / own window."""
    directions = [d for d in ('h2d', 'd2h') if any(d in w['bytes'] for w in windows)]
    return {d: sum(w['bytes'][d] / ((w['end_ns'] - w['start_ns']) / 1e9) for w in windows)
            for d in directions}


def worker(args):
    rank, world, node = args.rank, args.world, args.node
    if args.cpus:
        os.sched_setaffinity(0, parse_cpulist(args.cpus))  # threads created later inherit it
    if node is not None:
        bind_memory(node)  # before any allocation, so first touch lands on the GPU's node
    import torch
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{args.port}', rank=rank,
                            world_size=world)
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES holds exactly this process's GPU, by UUID
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if uuid.replace('GPU-', '') != args.uuid.replace('GPU-', ''):
        raise RuntimeError(f'expected GPU {args.uuid}, CUDA gave {uuid}')
    size, chunk = args.buffer_mib << 20, args.chunk_mib << 20
    host_src = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host_src.fill_(1)
    host_dst = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host_dst.fill_(0)
    dev_a = torch.zeros(size, dtype=torch.uint8, device='cuda')
    dev_b = torch.ones(size, dtype=torch.uint8, device='cuda')
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    buffers = (host_src, host_dst, dev_a, dev_b)
    torch.cuda.synchronize()
    for _ in range(3):  # untimed warm-up of both directions
        dev_a.copy_(host_src, non_blocking=True)
        host_dst.copy_(dev_b, non_blocking=True)
    torch.cuda.synchronize()
    pages = [pages_per_node(host_src), pages_per_node(host_dst)]
    me = {'rank': rank, 'gpu_uuid': uuid, 'memory_node': node,
          'affinity': sorted(os.sched_getaffinity(0)), 'buffer_pages': pages}
    directions = {}
    for direction in ('h2d', 'bi'):
        attempts = []
        for _ in range(3):
            dist.barrier()
            mine = window(torch, direction, buffers, streams, args.seconds, chunk, args.depth)
            everyone = [None] * world
            dist.all_gather_object(everyone, mine)
            ok, coverage = overlap(everyone, args.overlap)
            attempts.append({'windows': everyone, 'coverage': coverage, 'overlap_ok': ok})
            if ok:
                break
        directions[direction] = {
            'aggregate_gbs': {d: v / 1e9 for d, v in aggregates(everyone).items()} if ok else None,
            'attempts': attempts}
    procs = [None] * world
    dist.all_gather_object(procs, me)
    if rank == 0:
        print(json.dumps({'processes': procs, 'directions': directions}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


# ---------------------------------------------------------------- launcher

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def placement(result, bound):
    """'ok', 'off-node' or 'not checked' for one set and round."""
    if not bound:
        return 'not checked'
    for p in result['processes']:
        # Node keys come back through JSON as strings.
        if any(set(pages) != {str(p['memory_node'])} for pages in p['buffer_pages']):
            return 'off-node'
    return 'ok'


def run_set(gpu_set, info, topo, args, env):
    port = free_port()
    procs = []
    for rank, gpu in enumerate(gpu_set):
        node, cpus = topo[gpu]
        argv = [sys.executable, os.path.abspath(__file__), '--worker', '--rank', str(rank),
                '--world', str(len(gpu_set)), '--port', str(port), '--uuid', info[gpu]['uuid'],
                '--seconds', str(args.seconds), '--buffer-mib', str(args.buffer_mib),
                '--chunk-mib', str(args.chunk_mib), '--depth', str(args.depth),
                '--overlap', str(args.overlap)]
        if node is not None:
            argv += ['--node', str(node), '--cpus', cpus]
        child = dict(env, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=info[gpu]['uuid'])
        out = tempfile.TemporaryFile(mode='w+')  # a pipe could fill up and block the worker
        procs.append((subprocess.Popen(argv, env=child, stdout=out, text=True), out))
    # Two directions, up to three attempts each, plus start-up and warm-up.
    deadline = time.monotonic() + 6 * args.seconds + 180
    workers = [w for w, _ in procs]
    while any(w.poll() is None for w in workers):
        failed = [w for w in workers if w.poll() not in (None, 0)]
        if failed or time.monotonic() > deadline:
            for w in workers:
                w.kill()
            for w in workers:
                w.wait()
            for _, f in procs:
                f.close()
            return None, 'a worker failed' if failed else 'timed out'
        time.sleep(0.2)
    if any(w.returncode for w in workers):
        return None, 'a worker failed'
    out = procs[0][1]
    out.seek(0)
    lines = out.read().strip().splitlines()
    for _, f in procs:
        f.close()
    return json.loads(lines[-1]), None


def parse_sets(text, all_gpus):
    sets = []
    for item in filter(None, (s.strip() for s in text.split(';'))):
        try:
            s = list(all_gpus) if item == 'all' else [int(g) for g in item.split(',') if g.strip()]
        except ValueError:
            raise SystemExit(f'--sets: {item!r} is not a list of GPU indices')
        unknown = [g for g in s if g not in all_gpus]
        if unknown:
            raise SystemExit(f'--sets: no GPU {unknown}; this machine has {sorted(all_gpus)}')
        if len(set(s)) != len(s):
            raise SystemExit(f'--sets: GPU listed twice in {item!r}')
        sets.append(s)
    if not sets:
        raise SystemExit('--sets: no GPU set given')
    return sets


def versions():
    import torch
    driver = nvidia_smi('driver_version')[0][0]
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'driver': driver,
            'kernel': platform.release(), 'machine': platform.machine()}


def fmt(cell):
    return ' + '.join(f'{v:.2f}' for v in cell.values()) if cell else 'missing (overlap check failed)'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--sets', default='0', help='GPU sets separated by ";", e.g. "0;0,1;all"')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--seconds', type=float, default=3.0, help='copy window per direction')
    p.add_argument('--buffer-mib', type=int, default=256)
    p.add_argument('--chunk-mib', type=int, default=16)
    p.add_argument('--depth', type=int, default=2, help='copies in flight per stream')
    p.add_argument('--overlap', type=float, default=0.99, help='minimum common-window coverage')
    p.add_argument('--json', type=Path, help='write all rounds here (rewritten after every set)')
    p.add_argument('--allow-busy', action='store_true', help='run even if a GPU has other processes')
    for hidden in ('--rank', '--world', '--port', '--node'):
        p.add_argument(hidden, type=int, help=argparse.SUPPRESS)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--uuid', help=argparse.SUPPRESS)
    p.add_argument('--cpus', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args)

    if platform.machine() not in SYSCALLS:
        raise SystemExit(f'unsupported CPU architecture {platform.machine()} (x86_64 or aarch64)')
    if args.seconds <= 0 or args.rounds < 1 or args.depth < 1 or not 0 < args.overlap <= 1:
        raise SystemExit('--seconds, --rounds and --depth must be positive, --overlap in (0, 1]')
    if not 0 < args.chunk_mib <= args.buffer_mib:
        raise SystemExit('--chunk-mib must be positive and at most --buffer-mib')
    if args.json and not args.json.parent.is_dir():
        raise SystemExit(f'--json: directory {args.json.parent} does not exist')

    info = gpus()
    topo = {g: numa_of(v['bus_id']) for g, v in info.items()}
    sets = parse_sets(args.sets, sorted(info))
    nodes = numa_nodes()
    for g, v in info.items():
        print(f"GPU {g}: {v['name']}, {v['bus_id']}, NUMA node {topo[g][0]}, CPUs {topo[g][1]}")
    if len(nodes) > 1 and any(topo[g][0] is None for s in sets for g in s):
        print(f'warning: {len(nodes)} NUMA nodes, but some GPUs report none; their CPUs and '
              'buffers are not bound and placement is not checked')
    env = {k: v for k, v in os.environ.items() if k not in OWN_ENV}
    watched = {k: v for k, v in env.items() if k.startswith(WATCHED_ENV)}
    if watched:
        print('warning: inherited variables that can change copy behaviour: ' + ', '.join(sorted(watched)))

    record = {'args': {k: getattr(args, k) for k in ('sets', 'rounds', 'seconds', 'buffer_mib',
                                                      'chunk_mib', 'depth', 'overlap')},
              'versions': versions(), 'inherited_env': watched, 'numa_nodes': nodes,
              'gpus': {str(g): dict(v, numa_node=topo[g][0], cpus=topo[g][1]) for g, v in info.items()},
              'results': {','.join(map(str, s)): [] for s in sets}}
    for r in range(args.rounds):
        for s in sets:
            key = ','.join(map(str, s))
            busy = busy_gpus({info[g]['uuid'] for g in s})
            if busy and not args.allow_busy:
                raise SystemExit(f'GPU set {key} has other processes (use --allow-busy): {busy}')
            res, error = run_set(s, info, topo, args, env)
            if res is None:
                record['results'][key].append({'error': error})
                print(f'round {r + 1}  GPUs {key}: {error}', flush=True)
            else:
                bound = all(topo[g][0] is not None for g in s)
                res['placement'] = placement(res, bound)
                record['results'][key].append(res)
                d = res['directions']
                print(f"round {r + 1}  GPUs {key}: h2d {fmt(d['h2d']['aggregate_gbs'])}  "
                      f"bi {fmt(d['bi']['aggregate_gbs'])}  placement {res['placement']}", flush=True)
            if args.json:
                args.json.write_text(json.dumps(record, indent=1))

    print('\nmedian over rounds, GB/s (bi = host-to-device + device-to-host); '
          'rounds with off-node buffer pages are excluded')
    for key, results in record['results'].items():
        usable = [x for x in results if 'directions' in x and x['placement'] != 'off-node']
        cells = []
        for d in ('h2d', 'bi'):
            got = [x['directions'][d]['aggregate_gbs'] for x in usable if x['directions'][d]['aggregate_gbs']]
            if not got:
                cells.append(f'{d}: missing')
                continue
            meds = [statistics.median(g[k] for g in got) for k in ('h2d', 'd2h') if k in got[0]]
            cells.append(f'{d}: ' + ' + '.join(f'{m:.1f}' for m in meds) + f'  (n={len(got)})')
        print(f'  GPUs {key:<16} ' + '   '.join(cells))
    return 0


if __name__ == '__main__':
    sys.exit(main())
