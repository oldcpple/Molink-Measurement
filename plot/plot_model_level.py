import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# 设置专业风格的绘图参数
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'  # 使用无衬线字体
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = (14, 10)  # 增加高度以容纳新的事件
mpl.rcParams['figure.dpi'] = 100

# 专业配色方案（来自Tableau）
BATCH_COLORS = {
    0: '#4E79A7',  # 深蓝
    1: '#F28E2B',  # 橙色
    2: '#E15759',  # 红色
    3: '#76B7B2',  # 青绿
    4: '#59A14F',  # 绿色
    5: '#EDC948',  # 黄色
    6: '#B07AA1',  # 紫色
    7: '#FF9DA7',  # 粉红
    8: '#9C755F',  # 棕色
    9: '#BAB0AC',  # 灰色,
    10: '#499894', # 青色
    11: '#A25050'  # 深红
}

# 请求事件专用颜色
REQUEST_COLORS = {
    'added': '#2E8B57',  # 海绿色 - 请求到达
    'finished': '#DC143C'  # 深红色 - 请求完成
}

# Attn和MLP事件专用颜色
COMPONENT_COLORS = {
    'attn_start': '#FF6B6B',  # 红色 - attention开始
    'attn_end': '#FF6B6B',    # 红色 - attention结束
    'mlp_start': '#4ECDC4',   # 青色 - MLP开始
    'mlp_end': '#4ECDC4'      # 青色 - MLP结束
}

# 解析日志文件的函数 - 扩展以匹配attn和mlp事件
def parse_log_file(filename, log_type):
    events = defaultdict(list)
    request_events = defaultdict(list)  # 专门存储请求事件
    component_events = defaultdict(list)  # 存储attn和mlp事件
    
    with open(filename, 'r') as f:
        for line in f:
            # 首先尝试匹配请求事件
            request_added_match = re.match(r'request\s+([a-f0-9]+)\s+is added at\s+([\d.]+)$', line)
            request_finished_match = re.match(r'request\s+([a-f0-9]+)\s+finished at\s+([\d.]+)$', line)
            
            if request_added_match:
                request_id = request_added_match.group(1)
                timestamp = float(request_added_match.group(2))
                request_events[timestamp].append(('added', request_id))
                continue
            elif request_finished_match:
                request_id = request_finished_match.group(1)
                timestamp = float(request_finished_match.group(2))
                request_events[timestamp].append(('finished', request_id))
                continue
            
            # 匹配attn和mlp事件
            attn_start_match = re.match(r'attn starts at\s+([\d.]+)$', line)
            attn_end_match = re.match(r'attn ends at\s+([\d.]+)$', line)
            mlp_start_match = re.match(r'mlp starts at\s+([\d.]+)$', line)
            mlp_end_match = re.match(r'mlp ends at\s+([\d.]+)$', line)
            
            if attn_start_match:
                timestamp = float(attn_start_match.group(1))
                component_events['attn'].append(('start', timestamp))
                continue
            elif attn_end_match:
                timestamp = float(attn_end_match.group(1))
                component_events['attn'].append(('end', timestamp))
                continue
            elif mlp_start_match:
                timestamp = float(mlp_start_match.group(1))
                component_events['mlp'].append(('start', timestamp))
                continue
            elif mlp_end_match:
                timestamp = float(mlp_end_match.group(1))
                component_events['mlp'].append(('end', timestamp))
                continue
                
            # 匹配行首的数字（batch编号和可能的batch size）
            match = re.match(r'(\d+)(?:\s+(\d+))?\s+(.+?)\s+at\s+([\d.]+)$', line)
            if not match:
                continue
                
            batch_id = int(match.group(1))
            batch_size = match.group(2)
            event_desc = match.group(3).strip()
            timestamp = float(match.group(4))
            
            # 提取事件类型
            if "compute starts" in event_desc:
                event_type = 'compute_start'
                # 提取batch_size（如果存在）
                if batch_size:
                    batch_size = int(batch_size)
                else:
                    # 尝试从事件描述中提取
                    size_match = re.search(r'(\d+)\s+compute starts', event_desc)
                    if size_match:
                        batch_size = int(size_match.group(1))
            elif "compute ends" in event_desc:
                event_type = 'compute_end'
                batch_size = 0
            elif "trans starts" in event_desc:
                event_type = 'trans_start'
                batch_size = 0
            elif "trans ends" in event_desc:
                event_type = 'trans_end'
                batch_size = 0
            elif "serialization starts" in event_desc:
                event_type = 'serial_start'
                batch_size = 0
            elif "serialization ends" in event_desc:
                event_type = 'serial_end'
                batch_size = 0
            elif "back to head" in event_desc:
                event_type = 'back_to_head'
                batch_size = 0
            elif "recv" in event_desc:
                event_type = 'recv'
                batch_size = 0
            else:
                continue  # 忽略未知事件类型
            
            # 添加到事件列表
            events[batch_id].append((event_type, timestamp, batch_size or 0))
            
    return events, request_events, component_events

# 解析日志文件
print("解析server1日志...")
server1_events, server1_requests, server1_components = parse_log_file('llama2-7b,2080ti-2,1gbps,10ms,lambda=1,60s,server1.log', 'server1')
print("解析server2日志...")
server2_events, server2_requests, server2_components = parse_log_file('llama2-7b,2080ti-2,1gbps,10ms,lambda=1,60s,server2.log', 'server2')

# 提取所有时间戳用于归一化
all_timestamps = []
for batch_id, events in server1_events.items():
    print(f"Server1 Batch {batch_id} 事件:")
    for event_type, ts, size in events:
        print(f"  {event_type} at {ts}")
        all_timestamps.append(ts)

for batch_id, events in server2_events.items():
    print(f"Server2 Batch {batch_id} 事件:")
    for event_type, ts, size in events:
        print(f"  {event_type} at {ts}")
        all_timestamps.append(ts)

# 添加请求事件的时间戳
for ts, events in server1_requests.items():
    for event_type, req_id in events:
        print(f"Server1 Request {event_type}: {req_id} at {ts}")
        all_timestamps.append(ts)

# 添加组件事件的时间戳
for component, events in server1_components.items():
    for event_type, ts in events:
        print(f"Server1 {component} {event_type} at {ts}")
        all_timestamps.append(ts)

for component, events in server2_components.items():
    for event_type, ts in events:
        print(f"Server2 {component} {event_type} at {ts}")
        all_timestamps.append(ts)

if not all_timestamps:
    print("错误: 没有解析到任何事件!")
    exit(1)

base_time = min(all_timestamps)
print(f"基准时间: {base_time}")

# 归一化时间戳函数
def normalize(ts):
    return ts - base_time

# 创建绘图
fig, ax = plt.subplots(figsize=(16, 12))  # 增加高度
fig.set_facecolor('#F8F9FA')  # 浅灰色背景
ax.set_facecolor('#FFFFFF')   # 白色轴背景

# 资源层级定义 - 增加间距以容纳组件事件
RESOURCE_LEVELS = {
    'Server1 Compute': 5.0,
    'Server1 Components': 4.5,  # Server1的attn/mlp事件
    'Network Transfer': 3.0,
    'Server2 Compute': 2.0,
    'Server2 Components': 1.5   # Server2的attn/mlp事件
}

# 线宽和透明度设置
COMPUTE_LINEWIDTH = 16
COMPONENT_LINEWIDTH = 10  # 组件事件线宽
TRANSFER_LINEWIDTH = 12
ALPHA = 0.9

# 设置要显示的时间范围（使用原始时间戳）
# 根据您提供的数据调整
start_time = 1760946038.4348006
end_time = 1760946038.5348006

# 归一化时间范围
norm_start = normalize(start_time)
norm_end = normalize(end_time)
print(f"显示时间范围: {norm_start:.6f} 到 {norm_end:.6f} (相对基准时间)")

# 存储批注对象用于调整位置
annotations = []
legend_handles = []  # 用于自定义图例
found_transfers = []  # 存储找到的传输事件

# 绘制请求事件函数
def draw_request_events(request_events):
    request_counts = defaultdict(lambda: defaultdict(int))  # 按时间戳和事件类型计数
    
    # 统计每个时间戳的事件数量
    for ts, events in request_events.items():
        for event_type, req_id in events:
            request_counts[ts][event_type] += 1
    
    # 绘制事件
    for ts, event_counts in request_counts.items():
        norm_ts = normalize(ts)
        
        # 检查是否在时间范围内
        if norm_start <= norm_ts <= norm_end:
            for event_type, count in event_counts.items():
                color = REQUEST_COLORS[event_type]
                event_name = "arrives" if event_type == "added" else "finishes"
                
                # 绘制箭头指向Server1 Compute
                arrow = FancyArrowPatch(
                    (norm_ts, RESOURCE_LEVELS['Request Events']), 
                    (norm_ts, RESOURCE_LEVELS['Server1 Compute'] + 0.2),
                    arrowstyle='->', mutation_scale=15, color=color, linewidth=2, alpha=0.8
                )
                ax.add_patch(arrow)
                
                # 添加文本标注
                text = f"{count} req {event_name}"
                ann = ax.text(
                    norm_ts, RESOURCE_LEVELS['Request Events'] + 0.15, text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, boxstyle='round,pad=0.3')
                )
                annotations.append(ann)
                
                # 添加图例句柄（只添加一次）
                if event_type not in [h[0] for h in legend_handles if isinstance(h[0], str)]:
                    legend_handles.append((event_type, plt.Line2D([0], [0], color=color, lw=3, marker='o', markersize=8)))

# 绘制计算段函数 - 修改Server2的标注位置
def draw_compute_segments(events, resource_level, is_server1=True):
    for batch_id, batch_events in events.items():
        # 收集所有计算开始和结束事件
        compute_starts = [(ts, size) for etype, ts, size in batch_events if etype == 'compute_start']
        compute_ends = [ts for etype, ts, _ in batch_events if etype == 'compute_end']
        
        # 按时间排序以确保正确匹配
        compute_starts.sort(key=lambda x: x[0])
        compute_ends.sort()
        
        # 确保有匹配的事件对
        if len(compute_starts) != len(compute_ends):
            print(f"警告: Batch {batch_id} 的计算开始({len(compute_starts)})和结束({len(compute_ends)})事件数量不匹配")
            # 取最小数量进行匹配
            min_count = min(len(compute_starts), len(compute_ends))
            compute_starts = compute_starts[:min_count]
            compute_ends = compute_ends[:min_count]
        
        for (start, batch_size), end in zip(compute_starts, compute_ends):
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            # 检查是否在时间范围内
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                # 计算实际显示的起始点和结束点
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                # 计算持续时间（毫秒）
                duration_ms = (end - start) * 1000
                
                # 选择颜色
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                
                # 绘制计算线段
                line = ax.hlines(
                    y=resource_level, 
                    xmin=display_start, 
                    xmax=display_end,
                    colors=color,
                    linewidth=COMPUTE_LINEWIDTH,
                    alpha=ALPHA
                )
                
                # 添加图例句柄（只添加一次）
                if batch_id not in [h[0] for h in legend_handles if isinstance(h[0], int)]:
                    legend_handles.append((batch_id, plt.Line2D([0], [0], color=color, lw=4, alpha=ALPHA)))
                
                # 添加文本标注（batch size和持续时间）
                mid_x = (display_start + display_end) / 2
                text = f"bs={batch_size}\n{duration_ms:.1f}ms"
                
                # Server2的标注放在下方，Server1的标注放在中间
                if is_server1:
                    text_y = resource_level
                else:
                    text_y = resource_level - 0.25  # Server2标注下移
                    
                ann = ax.text(
                    mid_x, text_y, text,
                    ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                )
                annotations.append(ann)

# 绘制组件事件函数 (attn和mlp)
def draw_component_events(component_events, resource_level, server_name=""):
    component_legend_added = set()
    
    # 处理attn事件
    if 'attn' in component_events:
        attn_events = component_events['attn']
        starts = [ts for etype, ts in attn_events if etype == 'start']
        ends = [ts for etype, ts in attn_events if etype == 'end']
        
        starts.sort()
        ends.sort()
        
        # 匹配开始和结束事件
        for i in range(min(len(starts), len(ends))):
            start = starts[i]
            end = ends[i]
            
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            # 检查是否在时间范围内
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                # 计算持续时间（毫秒）
                duration_ms = (end - start) * 1000
                
                # 绘制attn线段
                ax.hlines(
                    y=resource_level, 
                    xmin=display_start, 
                    xmax=display_end,
                    colors=COMPONENT_COLORS['attn_start'],
                    linewidth=COMPONENT_LINEWIDTH,
                    alpha=ALPHA
                )
                
                # 添加文本标注
                mid_x = (display_start + display_end) / 2
                text = f"attn\n{duration_ms:.1f}ms"
                ann = ax.text(
                    mid_x, resource_level + 0.15, text,  # 标注放在线段上方
                    ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=COMPONENT_COLORS['attn_start'], boxstyle='round,pad=0.2')
                )
                annotations.append(ann)
                
                # 添加图例句柄
                if 'attn' not in component_legend_added:
                    legend_handles.append(('attn', plt.Line2D([0], [0], color=COMPONENT_COLORS['attn_start'], lw=3)))
                    component_legend_added.add('attn')
    
    # 处理mlp事件
    if 'mlp' in component_events:
        mlp_events = component_events['mlp']
        starts = [ts for etype, ts in mlp_events if etype == 'start']
        ends = [ts for etype, ts in mlp_events if etype == 'end']
        
        starts.sort()
        ends.sort()
        
        # 匹配开始和结束事件
        for i in range(min(len(starts), len(ends))):
            start = starts[i]
            end = ends[i]
            
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            # 检查是否在时间范围内
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                # 计算持续时间（毫秒）
                duration_ms = (end - start) * 1000
                
                # 绘制mlp线段
                ax.hlines(
                    y=resource_level - 0.15,  # 与attn稍微错开
                    xmin=display_start, 
                    xmax=display_end,
                    colors=COMPONENT_COLORS['mlp_start'],
                    linewidth=COMPONENT_LINEWIDTH,
                    alpha=ALPHA
                )
                
                # 添加文本标注
                mid_x = (display_start + display_end) / 2
                text = f"mlp\n{duration_ms:.1f}ms"
                ann = ax.text(
                    mid_x, resource_level - 0.3, text,  # 标注放在线段下方
                    ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=COMPONENT_COLORS['mlp_start'], boxstyle='round,pad=0.2')
                )
                annotations.append(ann)
                
                # 添加图例句柄
                if 'mlp' not in component_legend_added:
                    legend_handles.append(('mlp', plt.Line2D([0], [0], color=COMPONENT_COLORS['mlp_start'], lw=3)))
                    component_legend_added.add('mlp')

# 绘制传输段函数
def draw_transfer_segments():
    global found_transfers
    
    print("\n绘制中间结果传输 (server1 -> server2): trans_start to recv")
    for batch_id in set(list(server1_events.keys()) + list(server2_events.keys())):
        # 获取server1的trans_start事件
        s1_trans_starts = [ts for etype, ts, _ in server1_events.get(batch_id, []) 
                          if etype == 'trans_start']
        s1_trans_starts.sort()
        
        # 获取server2的recv事件
        s2_recvs = [ts for etype, ts, _ in server2_events.get(batch_id, []) 
                   if etype == 'recv']
        s2_recvs.sort()
        
        # 确保有匹配的事件
        if s1_trans_starts and s2_recvs:
            # 按顺序匹配事件
            for i in range(min(len(s1_trans_starts), len(s2_recvs))):
                start = s1_trans_starts[i]
                end = s2_recvs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                # 检查是否在时间范围内
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    # 计算实际显示的起始点和结束点
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    # 计算持续时间（毫秒）
                    duration_ms = (end - start) * 1000
                    
                    # 选择颜色
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    # 绘制传输线段
                    ax.hlines(
                        y=RESOURCE_LEVELS['Network Transfer'], 
                        xmin=display_start, 
                        xmax=display_end,
                        colors=color,
                        linewidth=TRANSFER_LINEWIDTH,
                        alpha=ALPHA,
                        linestyle='-'
                    )
                    
                    # 添加文本标注
                    mid_x = (display_start + display_end) / 2
                    text = f"To Server2\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Transfer'] - 0.25
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("intermediate", batch_id, start, end))
    
    print("\n绘制结果返回传输 (server2 -> server1): trans_start to back_to_head")
    for batch_id in set(list(server1_events.keys()) + list(server2_events.keys())):
        # 获取server2的trans_start事件
        s2_trans_starts = [ts for etype, ts, _ in server2_events.get(batch_id, []) 
                          if etype == 'trans_start']
        s2_trans_starts.sort()
        
        # 获取server1的back_to_head事件
        s1_backs = [ts for etype, ts, _ in server1_events.get(batch_id, []) 
                   if etype == 'back_to_head']
        s1_backs.sort()
        
        # 确保有匹配的事件
        if s2_trans_starts and s1_backs:
            # 按顺序匹配事件
            for i in range(min(len(s2_trans_starts), len(s1_backs))):
                start = s2_trans_starts[i]
                end = s1_backs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                # 检查是否在时间范围内
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    # 计算实际显示的起始点和结束点
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    # 计算持续时间（毫秒）
                    duration_ms = (end - start) * 1000
                    
                    # 选择颜色
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    # 绘制传输线段
                    ax.hlines(
                        y=RESOURCE_LEVELS['Network Transfer'], 
                        xmin=display_start, 
                        xmax=display_end,
                        colors=color,
                        linewidth=TRANSFER_LINEWIDTH,
                        alpha=ALPHA,
                        linestyle='-'
                    )
                    
                    # 添加文本标注
                    mid_x = (display_start + display_end) / 2
                    text = f"To Server1\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Transfer'] - 0.25
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("result", batch_id, start, end))

# 绘制所有部分
print("\n绘制Server1计算段...")
draw_compute_segments(server1_events, RESOURCE_LEVELS['Server1 Compute'])
print("绘制Server2计算段...")
draw_compute_segments(server2_events, RESOURCE_LEVELS['Server2 Compute'], is_server1=False)
print("绘制Server1组件事件...")
draw_component_events(server1_components, RESOURCE_LEVELS['Server1 Components'], "Server1")
print("绘制Server2组件事件...")
draw_component_events(server2_components, RESOURCE_LEVELS['Server2 Components'], "Server2")
print("绘制网络传输段...")
draw_transfer_segments()
print("绘制请求事件...")
draw_request_events(server1_requests)

if not found_transfers:
    print("警告: 没有找到任何网络传输事件!")
else:
    print(f"成功绘制 {len(found_transfers)} 个传输事件")

# 设置图表属性
ax.set_yticks(list(RESOURCE_LEVELS.values()))
ax.set_yticklabels(list(RESOURCE_LEVELS.keys()), fontsize=12, fontweight='bold')
ax.set_ylabel('Resources', fontsize=12, fontweight='bold')
ax.set_xlabel(f'Time (seconds from base time {base_time:.6f})', fontsize=11)
ax.set_title(f'Distributed Computing Timeline with Component Events', fontsize=14, fontweight='bold', pad=15)

# 设置x轴范围
ax.set_xlim(norm_start, norm_end)

# 设置y轴范围以包含所有事件
ax.set_ylim(0.8, 6.5)

# 添加时间刻度线
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

# 添加资源分隔线
for y in RESOURCE_LEVELS.values():
    ax.axhline(y=y, color='gray', alpha=0.3, linewidth=0.5)

# 创建专业图例
if legend_handles:
    # 分离不同类型的图例
    batch_handles = [h[1] for h in legend_handles if isinstance(h[0], int)]
    batch_labels = [f'Batch {h[0]}' for h in legend_handles if isinstance(h[0], int)]
    
    request_handles = [h[1] for h in legend_handles if isinstance(h[0], str) and h[0] in ['added', 'finished']]
    request_labels = [f'Request {h[0].title()}' for h in legend_handles if isinstance(h[0], str) and h[0] in ['added', 'finished']]
    
    component_handles = [h[1] for h in legend_handles if isinstance(h[0], str) and h[0] in ['attn', 'mlp']]
    component_labels = [f'{h[0].upper()}' for h in legend_handles if isinstance(h[0], str) and h[0] in ['attn', 'mlp']]
    
    # Batch颜色图例
    if batch_handles:
        batch_legend = plt.legend(
            batch_handles,
            batch_labels,
            title='Batch ID',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(batch_legend)
    
    # 请求事件图例
    if request_handles:
        request_legend = plt.legend(
            request_handles,
            request_labels,
            title='Request Events',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.85),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(request_legend)
    
    # 组件事件图例
    if component_handles:
        component_legend = plt.legend(
            component_handles,
            component_labels,
            title='Component Events',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.75),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(component_legend)

# 添加时间范围标记
ax.text(
    0.5, -0.10, 
    f'Time Range: {start_time:.6f} - {end_time:.6f} | Base Time: {base_time:.6f}',
    transform=ax.transAxes,
    ha='center',
    fontsize=9,
    color='#555555'
)

# 添加边框
for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')
    spine.set_linewidth=0.8

# 调整标注位置避免重叠
def adjust_annotations(annotations, min_distance=0.02):
    """调整标注位置避免重叠"""
    # 按x坐标排序
    annotations.sort(key=lambda ann: ann.get_position()[0])
    
    # 检查并调整重叠
    for i in range(1, len(annotations)):
        prev = annotations[i-1]
        curr = annotations[i]
        prev_pos = prev.get_position()
        curr_pos = curr.get_position()
        
        # 检查x坐标是否太近
        if abs(curr_pos[0] - prev_pos[0]) < min_distance:
            # 垂直偏移调整
            offset = min_distance - abs(curr_pos[0] - prev_pos[0])
            new_y = curr_pos[1] - offset * 0.5
            curr.set_position((curr_pos[0], new_y))

# 应用标注调整
if annotations:
    adjust_annotations(annotations)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.95, left=0.08, right=0.95)

# 保存图像（可选）
# plt.savefig('distributed_timeline_with_components.pdf', bbox_inches='tight', dpi=300)
plt.show()