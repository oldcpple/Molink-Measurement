#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展版 window_stats（无 argparse，参数在脚本顶部设置）

功能：
- 读取 CSV（默认第一列为时间戳，整数秒），同时读取第3列（请求长度）和第4列（响应长度）。
- 仅当 start t 与 end t+window 都恰好存在于时间戳集合中时，才把该窗口计入统计。
- 对每个满足条件的窗口：
    - 统计 [t, t+window] 内的请求数（包含端点）
    - 计算该窗口内请求长度的平均值（忽略缺失/非数值）
    - 计算该窗口内响应长度的平均值（忽略缺失/非数值）
- 对每种窗口长度输出汇总：符合条件窗口数 / 最大请求数 / 平均请求数 / 请求长度平均值（窗口间的平均）/ 响应长度平均值（窗口间的平均）
"""

import numpy as np
import pandas as pd
import math
import sys

# --------------------------
# 用户配置（请在此处修改）
CSV_PATH = "BurstGPT_without_fails_1.csv"   # <- CSV 文件路径
TS_COL = 0        # 时间戳列索引（从0开始）
REQLEN_COL = 2    # 第3列（请求长度），索引从0开始 -> 2
RESPLEN_COL = 3   # 第4列（响应长度），索引从0开始 -> 3
HAS_HEADER = True # CSV 是否含表头
WINDOWS = [1, 3, 5, 10]
# 若希望时间戳保留为 int（截断小数）为 False；若希望四舍五入为 True
ROUND_TS_TO_INT = False
# --------------------------

def load_and_clean(path, ts_col, req_col, resp_col, has_header):
    """
    返回排序前未排序的 numpy arrays: ts(int64), req(float, NaN allowed), resp(float, NaN allowed)
    丢弃无法解析为时间戳的行；请求/响应若无法解析则为 NaN（但保留该行用于时间位置）
    """
    header = 0 if has_header else None
    try:
        df = pd.read_csv(path, header=header, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"读取 CSV 失败: {e}")

    # 检查列索引是否越界
    max_col = df.shape[1] - 1
    for c in (ts_col, req_col, resp_col):
        if not isinstance(c, int):
            raise ValueError("脚本目前仅支持使用整数列索引 (0-based)。")
        if c < 0 or c > max_col:
            raise ValueError(f"列索引 {c} 超出范围 (0..{max_col})")

    # 先按字符串读取并 strip
    df_cols = df.iloc[:, [ts_col, req_col, resp_col]].astype(str).apply(lambda s: s.str.strip())
    ts_series = pd.to_numeric(df_cols.iloc[:,0], errors='coerce')  # 时间戳，不能解析的为 NaN
    req_series = pd.to_numeric(df_cols.iloc[:,1], errors='coerce') # 请求长度，NaN 表示缺失
    resp_series = pd.to_numeric(df_cols.iloc[:,2], errors='coerce')# 响应长度，NaN 表示缺失

    total_rows = len(ts_series)
    n_bad_ts = int(ts_series.isna().sum())
    if n_bad_ts > 0:
        print(f"[warn] 有 {n_bad_ts}/{total_rows} 行时间戳无法解析，将被丢弃（示例最多5个）:")
        bad_idx = np.where(ts_series.isna())[0][:5].tolist()
        for i in bad_idx:
            row = df.iloc[i, :].to_dict()
            print(f"  行 {i}: {row}")

    # 保留时间戳合法的行
    valid_mask = ~ts_series.isna()
    ts_clean = ts_series[valid_mask].to_numpy(dtype=np.float64)  # 暂用 float，稍后转 int
    req_clean = req_series[valid_mask].to_numpy(dtype=np.float64) # NaN 保留
    resp_clean = resp_series[valid_mask].to_numpy(dtype=np.float64)

    # 将时间戳转成 int64（按配置选择截断或四舍五入）
    if ROUND_TS_TO_INT:
        ts_int = np.round(ts_clean).astype(np.int64)
    else:
        ts_int = ts_clean.astype(np.int64)

    return ts_int, req_clean, resp_clean

def compute_stats(ts, req, resp, windows):
    """
    ts: 1D int64 numpy array (timestamps, may have重复)
    req, resp: 1D float numpy arrays aligned with ts (NaN allowed)
    返回字典：每个窗口长度 -> 统计信息
    统计信息包含：
      - num_windows: 符合条件的窗口数
      - counts: 每个合法窗口的请求数列表
      - max: 最大请求数 (or nan)
      - mean: 平均请求数 (or nan)
      - per_window_req_mean: 每个合法窗口内的请求长度平均值列表 (NaN 表示窗口内无有效 req)
      - per_window_resp_mean: 每个合法窗口内的响应长度平均值列表 (NaN 表示窗口内无有效 resp)
      - overall_req_mean: 对 per_window_req_mean 做 nanmean（所有合法窗口的平均 of averages），或 nan
      - overall_resp_mean: 同上
    """
    results = {}
    if ts.size == 0:
        for w in windows:
            results[w] = {
                'num_windows': 0, 'counts': [], 'max': math.nan, 'mean': math.nan,
                'per_window_req_mean': [], 'per_window_resp_mean': [],
                'overall_req_mean': math.nan, 'overall_resp_mean': math.nan
            }
        return results

    # 先按时间排序，保持 req/resp 对齐
    order = np.argsort(ts, kind='stable')
    ts_sorted = ts[order]
    req_sorted = req[order]
    resp_sorted = resp[order]

    unique_ts = np.unique(ts_sorted)
    ts_set = set(unique_ts.tolist())

    for w in windows:
        counts = []
        per_req_means = []
        per_resp_means = []
        for t in unique_ts:
            if (t + w) in ts_set:
                left = np.searchsorted(ts_sorted, t, side='left')
                right = np.searchsorted(ts_sorted, t + w, side='right')
                c = int(right - left)
                counts.append(c)
                # 对该区间的 req/resp 求平均（忽略 NaN）
                window_req = req_sorted[left:right]
                window_resp = resp_sorted[left:right]
                # 使用 numpy.nanmean，如果全部为 NaN 会抛出 Warning 并返回 NaN
                try:
                    req_mean = float(np.nanmean(window_req)) if window_req.size > 0 else float('nan')
                except Warning:
                    req_mean = float('nan')
                try:
                    resp_mean = float(np.nanmean(window_resp)) if window_resp.size > 0 else float('nan')
                except Warning:
                    resp_mean = float('nan')
                per_req_means.append(req_mean)
                per_resp_means.append(resp_mean)
        if len(counts) == 0:
            results[w] = {
                'num_windows': 0, 'counts': [], 'max': math.nan, 'mean': math.nan,
                'per_window_req_mean': [], 'per_window_resp_mean': [],
                'overall_req_mean': math.nan, 'overall_resp_mean': math.nan
            }
        else:
            counts_arr = np.array(counts, dtype=np.int64)
            # overall average of per-window means: 忽略 NaN 值
            overall_req_mean = float(np.nanmean(np.array(per_req_means, dtype=np.float64)))
            overall_resp_mean = float(np.nanmean(np.array(per_resp_means, dtype=np.float64)))
            results[w] = {
                'num_windows': len(counts),
                'counts': counts,
                'max': int(np.max(counts)),
                'mean': float(np.mean(counts)),
                'per_window_req_mean': per_req_means,
                'per_window_resp_mean': per_resp_means,
                'overall_req_mean': overall_req_mean,
                'overall_resp_mean': overall_resp_mean
            }
    return results

def print_results(results):
    print("\n统计结果：")
    print("窗口(s) | 符合窗口数 | 最大请求数 | 平均请求数 | 请求长度(窗口间平均) | 响应长度(窗口间平均)")
    print("-------------------------------------------------------------------------------------")
    for w in sorted(results.keys()):
        info = results[w]
        n = info['num_windows']
        mx = info['max']
        mean = info['mean']
        o_req = info['overall_req_mean']
        o_resp = info['overall_resp_mean']
        if math.isnan(mx):
            mx_s = "N/A"
            mean_s = "N/A"
        else:
            mx_s = str(mx)
            mean_s = f"{mean:.4f}"
        o_req_s = "N/A" if (o_req is None or (isinstance(o_req, float) and math.isnan(o_req))) else f"{o_req:.4f}"
        o_resp_s = "N/A" if (o_resp is None or (isinstance(o_resp, float) and math.isnan(o_resp))) else f"{o_resp:.4f}"
        print(f"{w:7d} | {n:9d} | {mx_s:10} | {mean_s:11} | {o_req_s:18} | {o_resp_s:18}")
    print("-------------------------------------------------------------------------------------\n")

def main():
    print("读取 CSV：", CSV_PATH)
    try:
        ts, req, resp = load_and_clean(CSV_PATH, TS_COL, REQLEN_COL, RESPLEN_COL, HAS_HEADER)
    except Exception as e:
        print("读取/清洗数据失败：", e)
        sys.exit(1)

    print(f"有效时间戳行数: {len(ts)} (示例前20个: {np.sort(ts)[:20].tolist()})")
    results = compute_stats(ts, req, resp, WINDOWS)
    print_results(results)

if __name__ == "__main__":
    main()
