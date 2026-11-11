"""Exercise P2P subgroup failure, nonparticipant propagation and reuse with Gloo."""
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import timedelta
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

    timeout = timedelta(seconds=30)
    store = dist.TCPStore("127.0.0.1", port, None, False, timeout=timeout, use_libuv=False)
    options = dist.ProcessGroupGloo._Options()
    options._timeout = timeout
    options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    group = None
    if rank in (1, 2):
        group = dist.ProcessGroupGloo(dist.PrefixStore("selected", store), rank - 1, 2, options)

    ops = SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None, memory_allocated=lambda: 0,
        memory_reserved=lambda: 0, mem_get_info=lambda: (4096, 8192),
    )
    manager = SimpleNamespace(
        device_type="cpu", device_module=ops, supports_device_ipc=lambda: True,
        supports_device_p2p=lambda: True, ipc_collect=lambda: None,
    )

    class LocalIPC:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def export(self, buffer):
            self.buffer = buffer
            return "test handle"

        def attach(self, handle, device_id):
            assert handle == "test handle"
            return self.buffer

        def detach(self):
            pass

    ipc = LocalIPC()
    context = zmq.Context()
    context.setsockopt(zmq.RCVTIMEO, 10000)
    context.setsockopt(zmq.SNDTIMEO, 10000)
    context.setsockopt(zmq.LINGER, 0)

    def bind_socket():
        sock = context.socket(zmq.REQ)
        address = f"inproc://checkpoint-p2p-gloo-{uuid.uuid4().hex}"
        sock.bind(address)
        return sock, [("local", address)]

    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = rank
    ps._world_size = 3
    ps._gpu_count = 3
    ps._auto_pg = False
    ps._store = store
    ps._store_barrier_counter = 0
    ps._store_barrier_trash = []
    ps.device_manager = manager
    ps._current_global_parameter_metas = {0: object()}
    ps._local_rdma_devices = {}
    ps._remote_rdma_devices = {}
    ps._detect_bucket_size = lambda *args, **kwargs: (8, True)
    ps._bind_zmq_socket = bind_socket
    ps._copy_to_buffer = lambda name, bucket, buffer, *args: buffer.fill_(0)
    ps._logger_rank0 = lambda *args: None
    ps._p2p_store = SimpleNamespace(register_named_tensors=lambda *_: None,
                                     unregister_named_tensors=lambda *_: None)
    barrier = ps.store_based_barrier
    ps.store_based_barrier = lambda: barrier(timedelta(seconds=12))
    buckets = []
    for i in range(2):
        meta = ParameterMeta(name=f"w{i}", dtype=torch.float32, shape=torch.Size([1]), aligned_size=4)
        buckets.append((i + 1, i + 1, H2DBucket(size=4, ranges=[BucketRange(0, 0, 4)], items=[meta])))

    state = {"round": 0, "loaded": [], "worker_errors": [], "hooks": []}

    def load(weights):
        name = weights[0][0]
        state["loaded"].append((state["round"], name))
        if rank == 1 and state["round"] == 0 and name == "w1":
            raise RuntimeError("synthetic last bucket failure")

    def request_worker(paths):
        try:
            update_weights_from_ipc(context, dict(paths)["local"], 0, run=load,
                                    post_hook=lambda: state["hooks"].append(state["round"]))
        except RuntimeError as exc:
            state["worker_errors"].append((state["round"], str(exc)))

    outcomes = []
    try:
        with (
            patch("checkpoint_engine.ps.dist.is_initialized", return_value=True),
            patch("checkpoint_engine.ps.dist.new_group", return_value=group),
            patch("checkpoint_engine.ps.dist.destroy_process_group"),
            patch("checkpoint_engine.ps.dist.broadcast",
                  side_effect=lambda tensor, src, group: group.broadcast(tensor, src - 1).wait()),
            patch("checkpoint_engine.ps.dist.all_reduce",
                  side_effect=lambda tensor, op, group: group.allreduce(tensor, op).wait()),
            patch("checkpoint_engine.ps.dist.barrier", side_effect=lambda group: group.barrier().wait()),
            patch("checkpoint_engine.ps._gen_h2d_buckets", return_value=buckets),
            patch("checkpoint_engine.ps.build_ipc_handler", return_value=ipc),
            patch("checkpoint_engine.worker.DeviceManager", return_value=manager),
            patch("checkpoint_engine.worker._ipc_handler_for_handle", return_value=ipc),
        ):
            for phase in range(3):
                state["round"] = phase
                try:
                    ps.update("synthetic", request_worker, ranks=[1, 2])
                    outcomes.append("success")
                except Exception as exc:
                    outcomes.append(str(exc))
                if phase == 0:
                    assert "Failed to update weights" in outcomes[-1], outcomes
                else:
                    assert outcomes[-1] == "success", outcomes
        assert ps._store_barrier_counter == 3
        if rank == 0:
            assert not store.check(["parameter_server_barrier-1:update_failed"])
            assert not state["loaded"] and not state["hooks"]
        else:
            assert state["worker_errors"] == [(0, "Some workers failed to update weights")]
            assert state["hooks"] == [1, 2]
            assert state["loaded"] == [(i, name) for i in range(3) for name in ("w0", "w1")]
        print(f"rank={rank} outcomes={outcomes} hooks={state['hooks']} "
              f"barrier_generations={ps._store_barrier_counter}", flush=True)
    finally:
        context.term()


def run():
    print("Python", sys.version.split()[0], "torch", torch.__version__, "pyzmq", zmq.__version__, flush=True)
    print("Checkout", Path.cwd(), flush=True)
    store = dist.TCPStore("127.0.0.1", 0, 1, True, wait_for_workers=False, use_libuv=False)
    workers = []
    try:
        for rank in range(3):
            workers.append(subprocess.Popen([sys.executable, "-B", __file__, str(rank), str(store.port)]))
        codes = [worker.wait(timeout=90) for worker in workers]
        print("rank exit codes", codes, flush=True)
        assert codes == [0, 0, 0]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run()
    else:
        check_rank(int(sys.argv[1]), int(sys.argv[2]))
