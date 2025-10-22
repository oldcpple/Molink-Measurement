#!/usr/bin/env python3
# coding: utf-8

"""
burst_sender.py

保证在给定时间窗口内提交（启动）所有请求的脚本。
调度采用 threading.Thread，每个调度点立即启动一个线程去发请求，
线程内部可阻塞等待 HTTP 返回，但不会阻塞后续调度。
"""

import time
import json
import random
import threading
from typing import List, Tuple
import requests
from word_list import WORD_LIST  # 确保你有这个模块或替换为你自己单词列表

# --------------------- 配置区域 ---------------------
URL = "http://localhost:20000/generate"    # 目标接口
HEADERS = {"Content-Type": "application/json"}

# prompt / model 参数（统一对所有请求生效）
INPUT_LENGTH = 591      # prompt 的单词数（例如 20）
MAX_TOKENS = 72       # 放到请求体里的 max_tokens

# 请求调度参数
TOTAL_REQUESTS = 20   # 总请求数
WINDOW_SECONDS = 1   # 时间窗口（秒）
SCHEDULING_MODE = "uniform"   # "uniform" 或 "poisson"
JITTER = 0.0           # 对 uniform 模式可加小抖动（秒），例如 0.01

# 并发控制
# 当使用 threading 逐个启动线程时，理论上线程数会等于 TOTAL_REQUESTS。
# 如果 TOTAL_REQUESTS 很大（例如几千），请小心主机资源（文件描述符、TCP 口等）。
REQUEST_TIMEOUT = 3000   # requests 超时时间（秒）

# 重试/失败策略（简单）
RETRY_ON_ERROR = 0     # 失败时最多重试次数

# fire-and-forget 控制：如果为 False 则在调度完成后等待所有请求返回并打印统计；
# 如果为 True 则不等待（脚本会在调度完后直接退出，但注意：Python 退出会终止进程，后台线程也会被杀死）。
WAIT_FOR_COMPLETION = True
# ----------------------------------------------------

def build_prompt(word_list: List[str], length: int) -> str:
    """
    从 word_list 中随机组合得到 length 个单词的 prompt（单词之间以空格分隔）。
    如果 length > len(word_list)，允许重复抽取。
    """
    if length <= 0:
        return ""
    if length <= len(word_list):
        tokens = random.sample(word_list, length)
    else:
        tokens = []
        while len(tokens) < length:
            needed = length - len(tokens)
            pick = random.sample(word_list, min(needed, len(word_list)))
            tokens.extend(pick)
        tokens = tokens[:length]
    return " ".join(tokens)

def make_request(prompt: str, max_tokens: int, session: requests.Session, timeout: float):
    """
    用 session 发送请求，返回 dict: { 'status': 'ok'/'err', 'text'/ 'error', 'elapsed' }
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0
    }
    t0 = time.time()
    try:
        r = session.post(URL, headers=HEADERS, json=payload, timeout=timeout)
        elapsed = time.time() - t0
        if r.status_code == 200:
            return {"status": "ok", "text": r.text.strip(), "elapsed": elapsed, "code": r.status_code}
        else:
            return {"status": "err", "error": f"HTTP {r.status_code}: {r.text}", "elapsed": elapsed, "code": r.status_code}
    except Exception as e:
        elapsed = time.time() - t0
        return {"status": "err", "error": str(e), "elapsed": elapsed, "code": None}

def schedule_times(total_requests: int, window: float, mode: str = "uniform", jitter: float = 0.0) -> List[float]:
    """
    返回一个长度为 total_requests 的时间点列表（相对于 start_time 的秒偏移），都在 [0, window] 之内。
    mode: "uniform" 或 "poisson"
    uniform: 均匀分布（可加 jitter）
    poisson: 指数间隔（泊松到达），累计后如果超出 window，会把时间线缩放到 window 内以保证正好 total_requests 个点
    """
    if total_requests <= 0:
        return []
    if mode == "uniform":
        times = []
        interval = window / total_requests
        for i in range(total_requests):
            base = i * interval
            if jitter and interval > 0:
                dt = random.uniform(-jitter, jitter)
            else:
                dt = 0.0
            t = base + dt
            if t < 0: t = 0.0
            times.append(t)
        times = [min(max(0.0, t), window) for t in times]
        random.shuffle(times)  # 打散顺序以避免严格线性提交（可选）
        times.sort()
        return times
    elif mode == "poisson":
        lam = total_requests / window if window > 0 else total_requests
        times = []
        t = 0.0
        for _ in range(total_requests):
            inter = random.expovariate(lam)
            t += inter
            times.append(t)
        max_t = max(times)
        if max_t > 0:
            scale = window / max_t
            times = [t * scale for t in times]
        else:
            times = [0.0 for _ in times]
        times.sort()
        return times
    else:
        raise ValueError("mode must be 'uniform' or 'poisson'")

def _send_and_store(idx: int, prompt: str, max_tokens: int, session: requests.Session, timeout: float, retry: int, results: List[Tuple]):
    """
    真正执行请求并把结果写回 results（线程安全地通过索引写回）。
    支持简单重试。
    """
    attempt = 0
    last_err = None
    while attempt <= retry:
        res = make_request(prompt, max_tokens, session, timeout)
        if res["status"] == "ok":
            results[idx] = (res, prompt)
            return
        else:
            last_err = res
            attempt += 1
            time.sleep(0.1)
    # 全部重试失败
    results[idx] = (last_err, prompt)

def run_schedule_fire_and_forget(total_requests: int, window_seconds: float, mode: str, input_length: int, max_tokens: int,
                 word_list: List[str], retry_on_error: int, timeout: float, jitter: float,
                 wait_for_completion: bool = True):
    """
    使用 threading.Thread 按排定时间逐个启动线程发出请求，保证所有线程（即所有请求）在 window_seconds 内被启动（提交）。
    wait_for_completion: 是否在调度完后等待所有线程结束（默认 True）。
    """
    times = schedule_times(total_requests, window_seconds, mode, jitter)
    start_time = time.time()
    results = [None] * total_requests
    threads: List[threading.Thread] = []

    # 共享 Session（通常 requests.Session 在多数并发场景下能被安全使用）。
    # 若对线程安全非常敏感，可在每个线程中创建独立的 Session。
    session = requests.Session()

    print(f"调度模式: {mode}, 总请求: {total_requests}, 窗口: {window_seconds}s")
    print("首个请求将在 {:.3f}s 内提交（相对现在）".format(times[0] if times else 0.0))

    for i, t_off in enumerate(times):
        now = time.time()
        wait_for = start_time + t_off - now
        if wait_for > 0:
            time.sleep(wait_for)   # 在准确时间点启动线程
        # 在调度点马上生成 prompt 并启动线程去发请求（线程内部会阻塞等待响应，但不会影响后续调度）
        random_l = random.randint(5, 3000)
        prompt = build_prompt(word_list, random_l)
        th = threading.Thread(target=_send_and_store, args=(i, prompt, max_tokens, session, timeout, retry_on_error, results))
        th.daemon = True
        th.start()
        threads.append(th)

    total_schedule_time = time.time() - start_time
    print(f"所有请求已在 {total_schedule_time:.3f}s 内启动（应 ≤ 窗口 {window_seconds}s）")

    if wait_for_completion:
        # 等待所有线程结束（这一步是在窗口结束后进行，不会影响提交）
        for th in threads:
            th.join()
        # 简单统计
        ok_count = sum(1 for r in results if r and r[0].get("status") == "ok")
        err_count = total_requests - ok_count
        # 平均耗时仅计算存在记录的项，保护除以零
        elapsed_sum = sum(r[0].get("elapsed", 0.0) for r in results if r and r[0])
        counted = sum(1 for r in results if r and r[0])
        avg_time = (elapsed_sum / counted) if counted > 0 else 0.0

        print("\n===== 运行汇总 =====")
        print(f"总请求数: {total_requests}, 成功: {ok_count}, 失败: {err_count}")
        print(f"总耗时(含等待全部返回): {time.time() - start_time:.3f}s, 平均单请求耗时(仅计有返回的请求): {avg_time:.3f}s")
    else:
        print("调度已完成，未等待请求返回（fire-and-forget 模式）。")

    return results

if __name__ == "__main__":
    random.seed(42)

    # 调用调度函数
    results = run_schedule_fire_and_forget(
        total_requests=TOTAL_REQUESTS,
        window_seconds=WINDOW_SECONDS,
        mode=SCHEDULING_MODE,
        input_length=INPUT_LENGTH,
        max_tokens=MAX_TOKENS,
        word_list=WORD_LIST,
        retry_on_error=RETRY_ON_ERROR,
        timeout=REQUEST_TIMEOUT,
        jitter=JITTER,
        wait_for_completion=WAIT_FOR_COMPLETION
    )

    # 如果选择不等待结果并直接退出，results 里多数项可能为 None
    if WAIT_FOR_COMPLETION:
        # 举例：保存结果到文件（可选）
        try:
            with open("burst_results.json", "w", encoding="utf-8") as f:
                json.dump([{"status": r[0].get("status") if r else None, "elapsed": (r[0].get("elapsed") if r else None)} for r in results], f, ensure_ascii=False, indent=2)
            print("结果已写入 burst_results.json")
        except Exception as e:
            print("写入结果文件失败:", e)
