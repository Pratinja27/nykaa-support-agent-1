import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from config import GLOBAL_TIMEOUT_SEC, NODE_TIMEOUT_SEC


class NodeTimeout(Exception):
    pass


class GlobalTimeout(Exception):
    pass


def run_with_timeout(fn, seconds, error_cls):
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=seconds)
        except FutureTimeout:
            raise error_cls("call exceeded %.1fs" % seconds)


def slow_lookup():
    time.sleep(3)
    return {"ok": True}


def node_with_timeout():
    return run_with_timeout(slow_lookup, NODE_TIMEOUT_SEC, NodeTimeout)


def long_graph_run():
    time.sleep(0.8)
    time.sleep(0.8)
    time.sleep(0.8)
    return {"nodes": ["guard", "classify", "rag", "format"]}


def global_graph_timeout():
    return run_with_timeout(long_graph_run, GLOBAL_TIMEOUT_SEC, GlobalTimeout)


def run_timeout_demos():
    node = {"fired": False, "error": None}
    overall = {"fired": False, "error": None}
    try:
        node_with_timeout()
    except NodeTimeout as exc:
        node["fired"] = True
        node["error"] = str(exc)
    try:
        global_graph_timeout()
    except GlobalTimeout as exc:
        overall["fired"] = True
        overall["error"] = str(exc)
    return {"node_timeout": node, "global_timeout": overall}
