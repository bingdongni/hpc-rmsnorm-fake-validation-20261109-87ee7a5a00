"""Run from the checkpoint-engine checkout using a Python environment with its dependencies."""

import os
import subprocess
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import zmq


def check_rank(rank, port):
    sys.path.insert(0, os.getcwd())
    from checkpoint_engine.data_types import BucketRange, H2DBucket, ParameterMeta
    from checkpoint_engine.ps import ParameterServer
    from checkpoint_engine.worker import update_weights_from_ipc

    timeout = timedelta(seconds=20)
    store = dist.TCPStore("127.0.0.1", port, None, False, timeout=timeout, use_libuv=False)
    options = dist.ProcessGroupGloo._Options()
    options._timeout = timeout
    options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    pg = dist.ProcessGroupGloo(store, rank, 2, options)

    ops = SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None, memory_allocated=lambda: 0,
        memory_reserved=lambda: 0, mem_get_info=lambda: (4096, 8192),
    )
    manager = SimpleNamespace(
        device_type="cpu", device_module=ops, supports_device_ipc=lambda: True,
        ipc_collect=lambda: None,
    )

    class LocalIPC:
        def export(self, buffer):
            self.buffer = buffer
            return "test handle"

        def attach(self, handle, device_id):
            assert handle == "test handle"
            return self.buffer

        def detach(self):
            pass

    ipc = LocalIPC()
    ctx = zmq.Context()
    ctx.setsockopt(zmq.RCVTIMEO, 10000)
    ctx.setsockopt(zmq.SNDTIMEO, 10000)
    ctx.setsockopt(zmq.LINGER, 0)
    socket = ctx.socket(zmq.REQ)
    address = f"inproc://checkpoint-gloo-review-{uuid.uuid4().hex}"
    socket.bind(address)

    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = rank
    ps.device_manager = manager
    ps._current_global_parameter_metas = {0: object()}
    ps._local_rdma_devices = {}
    ps._remote_rdma_devices = {}
    ps._detect_bucket_size = lambda *args, **kwargs: (8, True)
    ps._bind_zmq_socket = lambda: (socket, [("local", address)])
    ps._copy_to_buffer = lambda name, bucket, buffer, *args: buffer.fill_(0)
    ps._logger_rank0 = lambda *args: None
    buckets = []
    for i in range(2):
        meta = ParameterMeta(
            name=f"w{i}", dtype=torch.float32, shape=torch.Size([1]), aligned_size=4,
        )
        buckets.append((0, 0, H2DBucket(size=4, ranges=[BucketRange(0, 0, 4)], items=[meta])))
    loaded, worker_errors, hooks = [], [], []

    def load(weights):
        name = weights[0][0]
        loaded.append(name)
        if rank == 1 and name == "w1":
            raise RuntimeError("synthetic load failure")

    def request_worker(paths):
        try:
            update_weights_from_ipc(
                ctx, dict(paths)["local"], 0, run=load, post_hook=lambda: hooks.append(True),
            )
        except RuntimeError as exc:
            worker_errors.append(str(exc))

    failure = None
    try:
        with (
            patch("checkpoint_engine.ps.dist.is_initialized", return_value=True),
            patch("checkpoint_engine.ps.dist.broadcast",
                  side_effect=lambda tensor, src, group: pg.broadcast(tensor, src).wait()),
            patch("checkpoint_engine.ps.dist.all_reduce",
                  side_effect=lambda tensor, op, group: pg.allreduce(tensor, op).wait()),
            patch("checkpoint_engine.ps.dist.barrier", side_effect=lambda group: pg.barrier().wait()),
            patch("checkpoint_engine.ps._gen_h2d_buckets", return_value=buckets),
            patch("checkpoint_engine.worker.DeviceManager", return_value=manager),
            patch("checkpoint_engine.worker._ipc_handler_for_handle", return_value=ipc),
        ):
            try:
                ps._update_per_bucket("synthetic", request_worker, ipc, None)
            except RuntimeError as exc:
                failure = exc
    finally:
        socket.close(linger=0)
        ctx.term()
    print(f"rank={rank} injected_last_bucket_failure={rank == 1} "
          f"ps_error={type(failure).__name__}({failure}) loaded={loaded} "
          f"worker_errors={worker_errors} post_hook={bool(hooks)}", flush=True)
    assert str(failure) == "Failed to update weights due to remote errors"
    assert loaded == ["w0", "w1"]
    assert worker_errors == ["Some workers failed to update weights"]
    assert not hooks


def run():
    print("Python", sys.version.split()[0], "torch", torch.__version__, "pyzmq", zmq.__version__, flush=True)
    print("Checkout", Path.cwd(), flush=True)
    for args in (["rev-parse", "HEAD"], ["status", "--short"], ["diff", "--numstat"]):
        subprocess.run(["git", *args], check=True)
    for path in ("checkpoint_engine/ps.py", "tests/test_worker_update_errors.py"):
        if Path(path).is_file():
            oid = subprocess.check_output(["git", "hash-object", path], text=True).strip()
            print(f"{path} git-blob {oid}", flush=True)

    store = dist.TCPStore("127.0.0.1", 0, 1, True, wait_for_workers=False, use_libuv=False)
    workers = []
    try:
        for rank in range(2):
            workers.append(subprocess.Popen(
                [sys.executable, "-B", __file__, str(rank), str(store.port)],
            ))
        codes = [p.wait(timeout=60) for p in workers]
        print("rank exit codes", codes, flush=True)
        assert codes == [0, 0]
    finally:
        for p in workers:
            if p.poll() is None:
                p.kill()
                p.wait()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run()
    else:
        check_rank(int(sys.argv[1]), int(sys.argv[2]))
