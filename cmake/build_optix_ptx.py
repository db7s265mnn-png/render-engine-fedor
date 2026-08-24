#!/usr/bin/env python3
"""Compile OptiX kernels to PTX and embed them, one kernel at a time.

Windows CI with Ninja never printed [15/16] after embedding shade_surface:
cmd.exe &&-chains buffer nvcc, and ninja can stall between custom commands.
This driver is one process, prints with flush, and writes ptx_steps.log.
"""
from __future__ import print_function

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Logger(object):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, msg):
        line = "%s  %s\n" % (utc_now(), msg)
        sys.stdout.write(line)
        sys.stdout.flush()
        self.fp.write(line)
        self.fp.flush()

    def write_raw(self, text):
        if not text:
            return
        if not text.endswith("\n"):
            text += "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        self.fp.write(text)
        self.fp.flush()

    def close(self):
        self.fp.close()


class Heartbeat(threading.Thread):
    def __init__(self, log, label, interval=15.0):
        super(Heartbeat, self).__init__(daemon=True)
        self.log = log
        self.label = label
        self.interval = interval
        self.stop_evt = threading.Event()
        self.t0 = time.time()

    def run(self):
        while not self.stop_evt.wait(self.interval):
            self.log.write(
                "heartbeat: %s still running (%.0fs)"
                % (self.label, time.time() - self.t0)
            )


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.call(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def run_cmd(log, cmd, timeout, label):
    log.write("exec %s" % " ".join(cmd))
    hb = Heartbeat(log, label)
    hb.start()
    popen_kw = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "universal_newlines": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kw["start_new_session"] = True
    t0 = time.time()
    proc = subprocess.Popen(cmd, **popen_kw)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        log.write("TIMEOUT after %ss: %s" % (timeout, label))
        kill_tree(proc)
        out, _ = proc.communicate()
    hb.stop_evt.set()
    hb.join(1.0)
    log.write_raw(out)
    elapsed = time.time() - t0
    if timed_out:
        log.write("FAILED timeout %s (%.1fs)" % (label, elapsed))
        raise SystemExit(1)
    if proc.returncode != 0:
        log.write("FAILED rc=%s %s (%.1fs)" % (proc.returncode, label, elapsed))
        raise SystemExit(proc.returncode or 1)
    log.write("ok %s (%.1fs)" % (label, elapsed))


def main(argv):
    parser = argparse.ArgumentParser(description="Sequential OptiX PTX compile + embed")
    parser.add_argument("--manifest", required=True, help="JSON job list from CMake")
    parser.add_argument("--stamp", default="", help="Optional stamp file written on success")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-kernel nvcc timeout in seconds (0 = manifest / 600)",
    )
    args = parser.parse_args(argv[1:])

    # Line-buffer stdout even when ninja pipes us.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    log = Logger(manifest["log"])
    timeout = args.timeout or int(manifest.get("timeout_sec", 600))
    nvcc = manifest["nvcc"]
    python = manifest["python"]
    embed_script = manifest["embed_script"]
    common = list(manifest.get("common_flags") or [])
    kernels = list(manifest.get("kernels") or [])

    log.write("OptiX PTX driver start")
    log.write("python %s (%s)" % (python, sys.version.split()[0]))
    log.write("nvcc %s" % nvcc)
    log.write("kernels %s" % ", ".join(k["name"] for k in kernels))
    log.write("per-kernel timeout %ss" % timeout)

    t_all = time.time()
    try:
        for i, k in enumerate(kernels, 1):
            name = k["name"]
            source = k["source"]
            ptx = k["ptx"]
            embed = k["embed"]
            symbol = k["symbol"]
            extra = [f for f in (k.get("extra_flags") or []) if f]
            Path(ptx).parent.mkdir(parents=True, exist_ok=True)
            log.write("[%s/%s] nvcc %s" % (i, len(kernels), name))
            cmd = [nvcc] + common + extra + ["-o", ptx, source]
            run_cmd(log, cmd, timeout, "nvcc %s" % name)
            if not Path(ptx).is_file():
                log.write("FAILED missing PTX %s" % ptx)
                raise SystemExit(1)
            log.write("[%s/%s] embed %s (%s bytes)" % (i, len(kernels), name, Path(ptx).stat().st_size))
            run_cmd(
                log,
                [python, embed_script, ptx, embed, symbol],
                timeout,
                "embed %s" % name,
            )
            if not Path(embed).is_file():
                log.write("FAILED missing embed %s" % embed)
                raise SystemExit(1)
        if args.stamp:
            Path(args.stamp).write_text(utc_now() + "\n", encoding="utf-8")
        log.write("OptiX PTX driver done (%.1fs)" % (time.time() - t_all))
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
