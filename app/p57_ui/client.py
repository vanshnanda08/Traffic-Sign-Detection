"""Synchronous JSON subprocess, isolated temporary cache and real cleanup exit."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time


class WorkerFailure(RuntimeError):
    def __init__(self, message, receipt):
        self.receipt = receipt
        super().__init__(message+f" (exit {receipt['actual_exit']}; {receipt['elapsed_seconds_including_cleanup']:.2f}s including cleanup)")


def python_in(venv):
    return Path(venv) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def stop_and_collect(process):
    """Bounded graceful stop, then kill and reap; errors are never called success."""
    if process.poll() is None:
        try:process.terminate()
        except ProcessLookupError:pass
    try:return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        try:process.kill()
        except ProcessLookupError:pass
        return process.communicate()


def call_worker(root, request, timeout=240):
    root = Path(root).resolve()
    interpreter = Path(os.environ.get('P57_INFERENCE_PYTHON', python_in(root/'.venv')))
    if not interpreter.is_file():
        raise RuntimeError('Scientific interpreter missing. Install the pinned research environment; no automatic installation is performed.')
    payload = json.dumps(request, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix='p57-upload-') as temp:
        env = {**os.environ, 'PYTHONHASHSEED':'42', 'PYTORCH_ENABLE_MPS_FALLBACK':'0',
               'YOLO_AUTOINSTALL':'false', 'YOLO_CONFIG_DIR':str(Path(temp)/'ultralytics'),
               'MPLCONFIGDIR':str(Path(temp)/'matplotlib'), 'PYTHONDONTWRITEBYTECODE':'1',
               'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}
        started, wall = time.monotonic(), time.time()
        process = subprocess.Popen([str(interpreter), str(root/'app/worker.py')], cwd=root,
                                   env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        samples, stop = [], threading.Event()
        def observe():
            try:
                import psutil
            except ImportError:
                return
            while not stop.is_set():
                sample={'monotonic_seconds':time.monotonic()-started,'unavailable':{}}
                getters={'rss_bytes':lambda:psutil.Process(process.pid).memory_info().rss,
                         'system_available_bytes':lambda:psutil.virtual_memory().available,
                         'global_swap_bytes':lambda:psutil.swap_memory().used}
                for name,getter in getters.items():
                    try:sample[name]=getter()
                    except (OSError,psutil.Error) as error:
                        sample[name]=None
                        sample['unavailable'][name]={'type':type(error).__name__,'message':str(error)}
                samples.append(sample)
                stop.wait(.2)
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        timed_out = False
        try:
            stdout, stderr = process.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = stop_and_collect(process)
        except BaseException:
            # Preserve the error; terminate/reap our worker before its caches disappear.
            stop_and_collect(process)
            raise
        finally:
            stop.set(); observer.join(timeout=2)
        duration, span = time.monotonic()-started, time.time()-wall
        receipt = {'actual_exit':process.returncode, 'elapsed_seconds_including_cleanup':duration,
                   'pid':process.pid, 'operation':request['op'],
                   'command':[str(interpreter),str(root/'app/worker.py')],
                   'wall_span_seconds':span, 'resource_observations':samples,
                   'client_supervision_revision':'w8d.1',
                   'resource_method':'Unavailable OS/psutil counters are null with field-specific error; other counters continue. 200ms observed RSS/system available/global swap, including cleanup; sampled values not guaranteed peaks; not additive or exclusively attributable.',
                   'owned_temporary_directory_removed_on_return':True,'stderr':stderr[-12000:]}
        if timed_out:
            raise WorkerFailure('Inference timed out; owned child terminated and reaped.',receipt)
        try:
            response = json.loads(stdout)
        except (ValueError, TypeError) as error:
            raise WorkerFailure('Inference returned an invalid JSON reply.',receipt) from error
        if process.returncode != 0 or not response.get('ok'):
            receipt['worker_seconds_before_cleanup'] = response.get('worker_seconds_before_cleanup')
            raise WorkerFailure(f"Inference failed: {response.get('error', 'No completed result.')}",receipt)
        result = response['result']
        result['process'] = {**receipt,'worker_seconds_before_cleanup':response['worker_seconds_before_cleanup']}
        return result
