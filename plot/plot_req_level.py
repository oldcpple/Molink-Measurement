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
mpl.rcParams['figure.figsize'] = (14, 6)
mpl.rcParams['figure.dpi'] = 100

# 专业配色方案
EVENT_COLORS = {
    'added': '#4E79A7',           # 深蓝 - 请求到达
    'scheduled': '#F28E2B',       # 橙色 - 调度
    'compute_start': '#E15759',   # 红色 - 开始计算
    'compute_end': '#59A14F',     # 绿色 - 结束计算
    'returns_to_head': '#76B7B2', # 青绿 - 返回头节点
    'first_token': '#B07AA1',     # 紫色 - 获取第一个token
    'finished': '#EDC948',        # 黄色 - 请求完成
    'arrives_at_worker': '#FF9DA7', # 粉红 - 到达worker
    'compute_segment': '#499894'  # 青色 - 计算时间段
}

# 配置参数 - 请根据需要修改这些值
TARGET_REQUEST_ID = "7a97cb319dde43fcbae1aa7d9ba3e39b"  # 要追踪的请求ID
TIME_RANGE_START = 1760946856.3409428                   # 开始时间
TIME_RANGE_END = 1760946857.379114                      # 结束时间

# 解析日志文件的函数 - 针对新格式，只提取特定请求的事件
def parse_single_request_logs(server1_file, server2_file, target_request_id):
    server1_events = []
    server2_events = []
    
    # 解析server1日志
    with open(server1_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 匹配请求到达事件
            added_match = re.match(r'request\s+([a-f0-9]+)\s+is added at\s+([\d.]+)$', line)
            if added_match and added_match.group(1) == target_request_id:
                timestamp = float(added_match.group(2))
                server1_events.append(('added', timestamp))
                continue
                
            # 匹配调度事件
            scheduled_match = re.match(r'request\s+([a-f0-9]+)\s+is scheduled to run at\s+([\d.]+)$', line)
            if scheduled_match and scheduled_match.group(1) == target_request_id:
                timestamp = float(scheduled_match.group(2))
                server1_events.append(('scheduled', timestamp))
                continue
                
            # 匹配开始计算事件
            compute_start_match = re.match(r'request\s+([a-f0-9]+)\s+starts to compute on worker at\s+([\d.]+)$', line)
            if compute_start_match and compute_start_match.group(1) == target_request_id:
                timestamp = float(compute_start_match.group(2))
                server1_events.append(('compute_start', timestamp))
                continue
                
            # 匹配结束计算事件
            compute_end_match = re.match(r'request\s+([a-f0-9]+)\s+finishes computing on worker at\s+([\d.]+)$', line)
            if compute_end_match and compute_end_match.group(1) == target_request_id:
                timestamp = float(compute_end_match.group(2))
                server1_events.append(('compute_end', timestamp))
                continue
                
            # 匹配返回头节点事件
            returns_match = re.match(r'request\s+([a-f0-9]+)\s+returns to head server at\s+([\d.]+)$', line)
            if returns_match and returns_match.group(1) == target_request_id:
                timestamp = float(returns_match.group(2))
                server1_events.append(('returns_to_head', timestamp))
                continue
                
            # 匹配获取第一个token事件
            first_token_match = re.match(r'request\s+([a-f0-9]+)\s+got its first token at\s+([\d.]+)$', line)
            if first_token_match and first_token_match.group(1) == target_request_id:
                timestamp = float(first_token_match.group(2))
                server1_events.append(('first_token', timestamp))
                continue
                
            # 匹配请求完成事件
            finished_match = re.match(r'request\s+([a-f0-9]+)\s+finished at\s+([\d.]+)$', line)
            if finished_match and finished_match.group(1) == target_request_id:
                timestamp = float(finished_match.group(2))
                server1_events.append(('finished', timestamp))
                continue
    
    # 解析server2日志
    with open(server2_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 匹配到达worker事件
            arrives_match = re.match(r'request\s+([a-f0-9]+)\s+arrives at worker at\s+([\d.]+)$', line)
            if arrives_match and arrives_match.group(1) == target_request_id:
                timestamp = float(arrives_match.group(2))
                server2_events.append(('arrives_at_worker', timestamp))
                continue
                
            # 匹配开始计算事件
            compute_start_match = re.match(r'request\s+([a-f0-9]+)\s+starts to compute on worker at\s+([\d.]+)$', line)
            if compute_start_match and compute_start_match.group(1) == target_request_id:
                timestamp = float(compute_start_match.group(2))
                server2_events.append(('compute_start', timestamp))
                continue
                
            # 匹配结束计算事件
            compute_end_match = re.match(r'request\s+([a-f0-9]+)\s+finishes computing on worker at\s+([\d.]+)$', line)
            if compute_end_match and compute_end_match.group(1) == target_request_id:
                timestamp = float(compute_end_match.group(2))
                server2_events.append(('compute_end', timestamp))
                continue
    
    # 按时间戳排序
    server1_events.sort(key=lambda x: x[1])
    server2_events.sort(key=lambda x: x[1])
    
    return server1_events, server2_events

# 解析日志文件
print(f"解析请求 {TARGET_REQUEST_ID} 的事件...")
server1_events, server2_events = parse_single_request_logs(
    'llama2-7b,2080ti-2,1gbps,10ms,lambda=5,60s,server1.log',  # 替换为您的server1日志文件路径
    'llama2-7b,2080ti-2,1gbps,10ms,lambda=5,60s,server2.log',  # 替换为您的server2日志文件路径
    TARGET_REQUEST_ID
)

# 打印解析到的事件
print(f"Server1 事件数量: {len(server1_events)}")
for event_type, timestamp in server1_events:
    print(f"  {event_type}: {timestamp}")

print(f"Server2 事件数量: {len(server2_events)}")
for event_type, timestamp in server2_events:
    print(f"  {event_type}: {timestamp}")

if not server1_events and not server2_events:
    print(f"错误: 没有找到请求 {TARGET_REQUEST_ID} 的任何事件!")
    exit(1)

# 提取该请求的所有时间戳
all_timestamps = [event[1] for event in server1_events] + [event[1] for event in server2_events]

# 使用配置的时间范围
base_time = TIME_RANGE_START
norm_start = 0  # 相对时间起点
norm_end = TIME_RANGE_END - TIME_RANGE_START

print(f"基准时间: {base_time}")
print(f"显示时间范围: {norm_start:.6f} 到 {norm_end:.6f} (相对基准时间)")

# 归一化时间戳函数
def normalize(ts):
    return ts - base_time

# 创建绘图
fig, ax = plt.subplots(figsize=(16, 6))
fig.set_facecolor('#F8F9FA')  # 浅灰色背景
ax.set_facecolor('#FFFFFF')   # 白色轴背景

# 资源层级定义
RESOURCE_LEVELS = {
    'Server1': 2.0,
    'Server2': 1.0
}

# 事件标注位置（在资源线上方）
EVENT_LEVEL_OFFSET = 0.2

# 计算线段高度
COMPUTE_LINEWIDTH = 12

# 存储批注对象
annotations = []
legend_handles = []

# 绘制计算时间段函数
def draw_compute_segments(events, resource_level, server_name):
    compute_starts = []
    compute_ends = []
    
    # 收集计算开始和结束事件
    for event_type, timestamp in events:
        if event_type == 'compute_start':
            compute_starts.append(timestamp)
        elif event_type == 'compute_end':
            compute_ends.append(timestamp)
    
    # 按时间排序并匹配
    compute_starts.sort()
    compute_ends.sort()
    
    # 确保有匹配的事件对
    if len(compute_starts) != len(compute_ends):
        print(f"警告: {server_name} 的计算开始({len(compute_starts)})和结束({len(compute_ends)})事件数量不匹配")
        min_count = min(len(compute_starts), len(compute_ends))
        compute_starts = compute_starts[:min_count]
        compute_ends = compute_ends[:min_count]
    
    for start, end in zip(compute_starts, compute_ends):
        norm_start_ts = normalize(start)
        norm_end_ts = normalize(end)
        
        # 检查是否在时间范围内
        if norm_end_ts < norm_start or norm_start_ts > norm_end:
            continue
            
        # 绘制计算线段
        ax.hlines(
            y=resource_level, 
            xmin=max(norm_start_ts, norm_start), 
            xmax=min(norm_end_ts, norm_end),
            colors=EVENT_COLORS['compute_segment'],
            linewidth=COMPUTE_LINEWIDTH,
            alpha=0.8
        )
        
        # 添加持续时间标注
        duration_ms = (end - start) * 1000
        mid_x = (norm_start_ts + norm_end_ts) / 2
        
        # 根据服务器决定标注位置
        if server_name == 'Server1':
            ann_y = resource_level - 0.15
        else:
            ann_y = resource_level + 0.15
            
        text = f"{duration_ms:.1f}ms"
        ann = ax.text(
            mid_x, ann_y, text,
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3')
        )
        annotations.append(ann)

# 绘制事件箭头函数
def draw_event_arrows(events, resource_level, server_name):
    for event_type, timestamp in events:
        norm_ts = normalize(timestamp)
        
        # 检查是否在时间范围内
        if norm_ts < norm_start or norm_ts > norm_end:
            continue
            
        # 确定箭头方向
        if event_type in ['compute_start', 'compute_end']:
            # 计算事件用双向箭头
            arrow = FancyArrowPatch(
                (norm_ts, resource_level + EVENT_LEVEL_OFFSET), (norm_ts, resource_level),
                arrowstyle='<->', mutation_scale=12, 
                color=EVENT_COLORS[event_type], linewidth=2, alpha=0.9
            )
        else:
            # 其他事件用向下箭头
            arrow = FancyArrowPatch(
                (norm_ts, resource_level + EVENT_LEVEL_OFFSET), (norm_ts, resource_level),
                arrowstyle='->', mutation_scale=15, 
                color=EVENT_COLORS[event_type], linewidth=2, alpha=0.9
            )
        
        ax.add_patch(arrow)
        
        # 添加事件名称标注
        event_names = {
            'added': 'Added',
            'scheduled': 'Scheduled',
            'compute_start': 'Compute Start',
            'compute_end': 'Compute End', 
            'returns_to_head': 'Returns to Head',
            'first_token': 'First Token',
            'finished': 'Finished',
            'arrives_at_worker': 'Arrives at Worker'
        }
        
        event_name = event_names.get(event_type, event_type)
        ann = ax.text(
            norm_ts, resource_level + EVENT_LEVEL_OFFSET + 0.1, event_name,
            ha='center', va='bottom', fontsize=9, fontweight='bold', 
            color=EVENT_COLORS[event_type],
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=EVENT_COLORS[event_type], boxstyle='round,pad=0.3')
        )
        annotations.append(ann)
        
        # 添加时间戳标注
        time_text = f"{timestamp:.6f}"
        ann_time = ax.text(
            norm_ts, resource_level - 0.3, time_text,
            ha='center', va='top', fontsize=8, color='#666666',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
        )
        annotations.append(ann_time)

# 绘制所有部分
print("\n绘制Server1计算段...")
draw_compute_segments(server1_events, RESOURCE_LEVELS['Server1'], 'Server1')
print("绘制Server2计算段...")  
draw_compute_segments(server2_events, RESOURCE_LEVELS['Server2'], 'Server2')

print("绘制Server1事件箭头...")
draw_event_arrows(server1_events, RESOURCE_LEVELS['Server1'], 'Server1')
print("绘制Server2事件箭头...")
draw_event_arrows(server2_events, RESOURCE_LEVELS['Server2'], 'Server2')

# 设置图表属性
ax.set_yticks(list(RESOURCE_LEVELS.values()))
ax.set_yticklabels(list(RESOURCE_LEVELS.keys()), fontsize=12, fontweight='bold')
ax.set_ylabel('Servers', fontsize=12, fontweight='bold')
ax.set_xlabel(f'Time (seconds from {base_time:.6f})', fontsize=11)

# 设置标题
ax.set_title(f'Request {TARGET_REQUEST_ID} Timeline', fontsize=14, fontweight='bold', pad=15)

# 设置坐标轴范围
ax.set_xlim(norm_start, norm_end)
ax.set_ylim(0.5, 2.8)  # 调整y轴范围以容纳标注

# 添加网格和时间刻度线
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

# 添加资源分隔线
for y in RESOURCE_LEVELS.values():
    ax.axhline(y=y, color='gray', alpha=0.5, linewidth=1)

# 创建图例
# 事件类型图例
event_legend_elements = []
for event_type, color in EVENT_COLORS.items():
    if event_type != 'compute_segment':  # 计算线段单独处理
        event_legend_elements.append(
            Line2D([0], [0], color=color, lw=2, marker='o', markersize=6, 
                   label=event_type.replace('_', ' ').title())
        )

# 计算线段图例
event_legend_elements.append(
    Line2D([0], [0], color=EVENT_COLORS['compute_segment'], lw=6, 
           label='Compute Segment')
)

# 添加图例
legend = ax.legend(
    handles=event_legend_elements,
    loc='upper right',
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    framealpha=0.9,
    edgecolor='#CCCCCC',
    title='Event Types'
)

# 添加总时间信息
total_time = norm_end - norm_start
ax.text(
    0.02, 0.98, 
    f'Displayed Time Range: {total_time:.3f}s',
    transform=ax.transAxes,
    va='top',
    fontsize=11,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
)

# 添加配置信息
config_info = f'Request ID: {TARGET_REQUEST_ID}\nTime Range: {TIME_RANGE_START:.6f} to {TIME_RANGE_END:.6f}'
ax.text(
    0.98, 0.98, 
    config_info,
    transform=ax.transAxes,
    va='top',
    ha='right',
    fontsize=9,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
)

# 调整标注位置避免重叠
def adjust_annotations(annotations, min_distance=0.05):
    """调整标注位置避免重叠"""
    annotations.sort(key=lambda ann: ann.get_position()[0])
    
    for i in range(1, len(annotations)):
        prev = annotations[i-1]
        curr = annotations[i]
        prev_pos = prev.get_position()
        curr_pos = curr.get_position()
        
        if abs(curr_pos[0] - prev_pos[0]) < min_distance:
            offset = min_distance - abs(curr_pos[0] - prev_pos[0])
            new_y = curr_pos[1] - offset * 0.3
            curr.set_position((curr_pos[0], new_y))

# 应用标注调整
if annotations:
    adjust_annotations(annotations)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.88, left=0.08, right=0.95)

# 保存图像（可选）
# plt.savefig(f'request_{TARGET_REQUEST_ID}_timeline.pdf', bbox_inches='tight', dpi=300)
plt.show()

print(f"\n可视化完成!")
print(f"请求ID: {TARGET_REQUEST_ID}")
print(f"显示时间范围: {total_time:.3f} 秒")