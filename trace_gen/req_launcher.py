#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
import time
import random
import sys
from typing import Optional
from sharegpt_loader import ShareGPTLoader

def start_poisson_for_duration(
    url: str,
    loader: ShareGPTLoader,
    avg_rps: float,
    duration_seconds: float,
    max_concurrency: Optional[int] = None,
    verbose: bool = True,
):
    """
    按泊松到达（指数分布间隔）在指定持续时间内提交请求。
    - url: curl 请求 URL（字符串）
    - data: 请求体（字典，会 json.dumps）
    - avg_rps: 每秒平均请求数 λ（requests/second），必须 > 0
    - duration_seconds: 持续时间（秒），脚本在发送到达持续时间到时立即返回（不等待子进程）
    - max_concurrency: 可选并发上限（同时存在的子进程数），None 表示不限制
    - verbose: 是否打印简短提交进度
    """
    if avg_rps <= 0:
        raise ValueError("avg_rps 必须 > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds 必须 > 0")
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError("max_concurrency 必须为正整数或 None")
    


    
    dn = subprocess.DEVNULL  # 跨平台的空输出

    start_time = time.time()
    end_time = start_time + duration_seconds
    submits = 0
    processes = []  # 存放当前仍未被清理的 Popen 对象

    try:
        while True:
            prompt = loader.get_random_qa().get('human')
            data = {
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0
                }
            data_str = json.dumps(data)
            now = time.time()
            if now >= end_time:
                break

            # 清理已经结束的子进程（非阻塞）
            if processes:
                alive = []
                for p in processes:
                    if p.poll() is None:
                        alive.append(p)
                processes = alive

            # 如果设置了 max_concurrency，且达到上限，则短暂等待并重试（不阻塞任何单个进程）
            if max_concurrency is not None and len(processes) >= max_concurrency:
                # 为避免忙等，睡一小段时间后继续检查
                # 这里睡眠一个很短的时间，既让出 CPU 又能及时恢复
                time.sleep(0.001)
                continue

            # 构造 curl 命令（-sS 保证静默但出错时仍能返回 stderr；但我们把 stderr 重定向到 DEVNULL）
            command = [
                "curl", "-sS", url,
                "-H", "Content-Type: application/json",
                "-d", data_str
            ]

            print(f'command is {command}')

            # 启动子进程，不等待完成，输出重定向到 DEVNULL
            try:
                p = subprocess.Popen(command, stdout=dn, stderr=dn, close_fds=True)
                processes.append(p)
                submits += 1
                if verbose:
                    elapsed = now - start_time
                    print(f"[提交] #{submits} (t={elapsed:.3f}s)  当前并发={len(processes)}")
            except Exception as e:
                # 启动失败则记录并继续（不阻塞）
                if verbose:
                    print(f"[ERROR] 启动请求失败: {e}", file=sys.stderr)

            # 计算下一到达间隔（指数分布），参数 λ = avg_rps
            # random.expovariate 的参数是 lambda（到达率）
            wait_seconds = random.expovariate(avg_rps)

            # 若下一个到达时间会超出结束时间，则不再等待并退出发送循环
            if time.time() + wait_seconds >= end_time:
                break

            # 为避免极短的 sleep 导致 busy-loop，我们对最小睡眠进行微小下限
            if wait_seconds < 0.0005:
                # 很短的等待直接微睡以减轻 CPU 占用
                time.sleep(0.0005)
            else:
                time.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("\n用户中断，停止提交。", file=sys.stderr)

    total_elapsed = time.time() - start_time
    # 最终不再等待子进程完成（按需可以改为等待）
    print(f"提交窗口已结束（实际提交耗时 {total_elapsed:.3f} 秒），共提交请求数: {submits}")
    print("脚本退出：不等待子进程完成。")

if __name__ == "__main__":



    default_url = "http://172.17.0.7:20000/generate"
    default_data = {
        "prompt": "San Francisco is a",
        "max_tokens": 50,
        "temperature": 0
    }

    try:
        print("按泊松到达（以 avg_rps 控制平均每秒请求数）持续提交请求脚本")

        url = default_url

        while True:
            s = input("每秒平均请求数 avg_rps（requests/second，>0，例如 5 或 0.5）: ").strip()
            try:
                avg_rps = float(s)
                if avg_rps <= 0:
                    raise ValueError
                break
            except Exception:
                print("请输入一个大于 0 的数值。")

        while True:
            s = input("持续时间（秒，例如 30）: ").strip()
            try:
                duration = float(s)
                if duration <= 0:
                    raise ValueError
                break
            except Exception:
                print("请输入一个大于 0 的数值（秒）。")

        s = False
        max_concurrency = None
        if s:
            try:
                m = int(s)
                if m <= 0:
                    raise ValueError
                max_concurrency = m
            except Exception:
                print("max_concurrency 非法，使用不限制。")
                max_concurrency = None

        s = 'y'
        verbose = s.startswith("y")

        loader = ShareGPTLoader("/mnt/lvm-data/home/dataset/sharegpt/common_en_70k.jsonl")
        print('正在加载shareGPT数据集到内存...')
        loader.load_data()
        print('加载完毕，即将开始测试')

        print("准备开始提交（按 Ctrl+C 可提前结束）...")
        start_poisson_for_duration(
            url=url,
            loader=loader,
            avg_rps=avg_rps,
            duration_seconds=duration,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )

    except KeyboardInterrupt:
        print("\n用户中断，退出。")
        sys.exit(0)
