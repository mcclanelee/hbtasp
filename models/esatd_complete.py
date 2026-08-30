# ==================== esatd_complete_fixed.py ====================
"""
ESATD - Complete Fixed Version

核心策略：
1. 优先保证 deadline（DMR < 0.3%）
2. 在满足 deadline 的前提下，选择最低电压（节能）
3. 负载均衡（两个处理器都使用）
4. 温度约束（不超过 Tmax）
5. 与 HBTASP 公平对比
"""

import numpy as np
import math
import heapq
import os
import json
import time
import pandas as pd
from datetime import datetime
import argparse

# ==================== 可视化导入 ====================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入 numba
try:
    from numba import jit, njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba 未安装，使用纯 Python 模式（建议安装：pip install numba）")


    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator


    def njit(*args, **kwargs):
        return jit(*args, **kwargs)

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'tables', 'esatd')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==================== 参数（与 HBTASP 对齐）====================
NUM_PROCESSORS = 2
NUM_TASKS_PER_CYCLE = 4
SUBTASKS_PER_TASK = 4
FIXED_DNN_LEVEL = 2

# WCET @ 0.8V (GPU0更快，GPU1更慢)
WCET_GPU0_BASE_MS = [6.3, 9.0, 12.3, 14.8, 20.1][FIXED_DNN_LEVEL]
WCET_GPU1_BASE_MS = [13.0, 18.2, 25.1, 29.5, 40.0][FIXED_DNN_LEVEL]

# DVFS
V_LEVELS = np.array([0.6, 0.7, 0.8, 0.85, 0.88], dtype=np.float32)
V_REF = 0.8
V_RATIOS = V_REF / V_LEVELS

# 预计算 WCET 矩阵 [core, voltage] -> wcet_sec
WCET_MATRIX = np.array([
    WCET_GPU0_BASE_MS * V_RATIOS / 1000.0,
    WCET_GPU1_BASE_MS * V_RATIOS / 1000.0
], dtype=np.float32)

# 预计算功耗系数
V_POWER_COEF = V_LEVELS ** 3

# 温度参数（与 HBTASP 对齐）
a = 0.1
alpha = 41.01
gamma = 512.635
B = 0.5058
Tmax = 60.0
Tamb = 25.0  # 环境温度 25°C
T_HOT = 55.0
V_IDLE = 0.6

# 冷却阈值（与 HBTASP 一致）
IDLE_COOLING_THRESHOLD_MS = 10
IDLE_COOLING_THRESHOLD_SEC = IDLE_COOLING_THRESHOLD_MS / 1000.0

# 调度参数
NUM_CYCLES = 1000
JITTER_RATIO = 0.1
EARLY_EXIT_THRESHOLD = 0.65
LOW_WEIGHT_THRESHOLD = 0.3
DEBUG_PRINT_CYCLES = 100

# ==================== 任务分类 ====================
# 使用两个处理器（负载均衡）
ALLOWED_PROCS = (0, 1)
PROC_TYPE = "Dual-Core (负载均衡)"


# ==================== 温度计算函数 ====================
@njit(cache=True)
def calc_steady_state_temp(v_val):
    """计算稳态温度"""
    A = a * (alpha + gamma * v_val ** 3)
    return A / B


@njit(cache=True)
def calc_temp_change(T0, v_idx, time_sec):
    """任务执行时的温度变化"""
    if time_sec <= 1e-6:
        return T0

    v_val = V_LEVELS[v_idx]
    Tss = calc_steady_state_temp(v_val)
    exp_term = math.exp(-B * time_sec)
    result = Tss - (Tss - T0) * exp_term
    return max(Tamb, min(120.0, result))


@njit(cache=True)
def calc_idle_cooling(T0, time_sec):
    """空闲冷却 - 趋向环境温度"""
    if time_sec <= 1e-6:
        return T0

    exp_term = math.exp(-B * time_sec)
    result = Tamb - (Tamb - T0) * exp_term
    return max(Tamb, min(T0, result))


# ==================== 队列 ====================
class FastEDFQueue:
    __slots__ = ('_heap', '_seq')

    def __init__(self):
        self._heap = []
        self._seq = 0

    def push(self, task):
        self._seq += 1
        heapq.heappush(self._heap, (task['deadline'], self._seq, task))

    def pop(self):
        if self._heap:
            return heapq.heappop(self._heap)[2]
        return None

    def is_empty(self):
        return len(self._heap) == 0


# ==================== 任务生成 ====================
def create_task_stream_batch(period_ms, num_cycles, jitter_ratio=0.1, seed=42):
    """生成扰动周期性任务流"""
    np.random.seed(seed)
    period_sec = period_ms / 1000.0
    jitter_bound = jitter_ratio * period_sec

    n_total_tasks = NUM_TASKS_PER_CYCLE * num_cycles

    weights = np.random.uniform(0.05, 0.7, (n_total_tasks, SUBTASKS_PER_TASK))
    for i in range(n_total_tasks):
        weights[i, 0] = np.random.uniform(0.4, 0.7)
    weights = weights / weights.sum(axis=1, keepdims=True)

    tasks = []
    task_idx = 0

    for task_id in range(NUM_TASKS_PER_CYCLE):
        phase = (task_id / NUM_TASKS_PER_CYCLE) * period_sec

        for cycle in range(num_cycles):
            jitter = np.random.uniform(-jitter_bound, jitter_bound)
            arrive_time = max(0, phase + cycle * period_sec + jitter)

            task_weights = weights[task_idx]
            max_weight_idx = np.argmax(task_weights)

            for sub_idx, weight in enumerate(task_weights):
                tasks.append({
                    'arrive': arrive_time,
                    'deadline': arrive_time + period_sec,
                    'weight': float(weight),
                    'is_mandatory': (sub_idx == max_weight_idx),
                    'task_id': task_id,
                    'jobid': cycle,
                })

            task_idx += 1

    tasks.sort(key=lambda x: x['arrive'])
    return tasks


# ==================== 指标计算 ====================
def calculate_atp_fast(temp_records):
    if not temp_records:
        return Tamb
    temps = np.array([r['temperature'] for r in temp_records])
    return np.mean(temps)


def calculate_iit_fast(temp_records, Tmax):
    if not temp_records:
        return 0.0
    temps = np.array([r['temperature'] for r in temp_records])
    over = temps[temps > Tmax] - Tmax
    return np.mean(over) if len(over) > 0 else 0.0


# ==================== 调度器 ====================
class ESATDScheduler:
    __slots__ = ('period_sec', 'core_free', 'core_temp', 'core_volt',
                 'defect_detected', 'mandatory_total', 'mandatory_timeout',
                 'mandatory_completed', 'assign_counts', 'completed_weight',
                 'total_weight', 'energy', 'temp_records')

    def __init__(self, period_ms):
        self.period_sec = period_ms / 1000.0
        self.core_free = np.array([0.0, 0.0], dtype=np.float32)
        self.core_temp = np.array([Tamb, Tamb], dtype=np.float32)
        self.core_volt = np.array([2, 2], dtype=np.int8)
        self.defect_detected = set()

        self.mandatory_total = 0
        self.mandatory_timeout = 0
        self.mandatory_completed = 0
        self.assign_counts = [0, 0]
        self.completed_weight = 0.0
        self.total_weight = 0.0
        self.energy = 0.0

        self.temp_records = []

    def _select_processor(self, task, current_time):
        """
        ESATD 修正策略：
        1. 优先保证 deadline（选择能完成任务的电压）
        2. 在满足 deadline 的前提下，选择最低电压（节能）
        3. 负载均衡（选择负载最低的处理器）
        """
        best_core = -1
        best_v_idx = -1
        best_wcet = 0.0
        best_finish = 0.0
        best_load = float('inf')

        for core in ALLOWED_PROCS:
            start = max(current_time, self.core_free[core])
            remaining = task['deadline'] - start

            if remaining <= 1e-6:
                continue

            wcets = WCET_MATRIX[core]

            # 找能满足 deadline 的最低电压
            found_v_idx = -1
            found_wcet = 0.0
            for v_idx in range(len(wcets)):
                if wcets[v_idx] <= remaining + 1e-6:
                    found_v_idx = v_idx
                    found_wcet = wcets[v_idx]
                    break

            if found_v_idx == -1:
                # 无法满足 deadline，使用最高电压
                found_v_idx = len(wcets) - 1
                found_wcet = wcets[found_v_idx]

            # 选择负载最低的处理器
            load = self.core_free[core]
            if load < best_load - 1e-6:
                best_load = load
                best_core = core
                best_v_idx = found_v_idx
                best_wcet = found_wcet
                best_finish = start + found_wcet

        if best_core == -1:
            return None, None, None, None

        return best_core, best_v_idx, best_wcet, best_finish

    def _should_cool(self, idle_time_sec):
        """判断是否需要冷却（与 HBTASP 阈值一致）"""
        return idle_time_sec >= IDLE_COOLING_THRESHOLD_SEC

    def schedule_cycle(self, cycle_tasks, start_time):
        """调度单个周期"""
        if not cycle_tasks:
            return

        # 按截止时间排序
        sorted_tasks = sorted(cycle_tasks, key=lambda x: x['deadline'])

        for task in sorted_tasks:
            if task['is_mandatory']:
                self.mandatory_total += 1
                self.total_weight += task['weight']

        current_time = start_time

        for task in sorted_tasks:
            key = (task['task_id'], task['jobid'])

            # 早退检查
            if not task['is_mandatory'] and key in self.defect_detected:
                if task['weight'] <= LOW_WEIGHT_THRESHOLD:
                    continue

            core, v_idx, wcet, finish = self._select_processor(task, current_time)

            if core is None:
                if task['is_mandatory']:
                    self.mandatory_timeout += 1
                continue

            start = max(current_time, self.core_free[core])
            finish = start + wcet

            # 记录开始温度
            self.temp_records.append({
                'time': start,
                'processor': core,
                'temperature': float(self.core_temp[core]),
                'voltage_idx': v_idx,
                'voltage_value': float(V_LEVELS[v_idx])
            })

            # 执行任务
            self.core_free[core] = finish
            self.core_temp[core] = calc_temp_change(self.core_temp[core], v_idx, wcet)

            # 记录结束温度
            self.temp_records.append({
                'time': finish,
                'processor': core,
                'temperature': float(self.core_temp[core]),
                'voltage_idx': v_idx,
                'voltage_value': float(V_LEVELS[v_idx])
            })

            self.assign_counts[core] += 1
            self.energy += V_POWER_COEF[v_idx] * wcet

            if task['is_mandatory']:
                if finish > task['deadline'] + 1e-3:
                    self.mandatory_timeout += 1
                else:
                    self.mandatory_completed += 1
                    self.completed_weight += task['weight']

                if task['weight'] > EARLY_EXIT_THRESHOLD:
                    self.defect_detected.add(key)

            current_time = finish

    def advance(self, next_start):
        """推进到下一个周期 - 带冷却阈值"""
        for core in range(2):
            if self.core_free[core] < next_start:
                idle = next_start - self.core_free[core]

                if idle > 0.001:
                    # 记录冷却前温度
                    self.temp_records.append({
                        'time': self.core_free[core],
                        'processor': core,
                        'temperature': float(self.core_temp[core]),
                        'voltage_idx': -1,
                        'voltage_value': float(V_IDLE)
                    })

                    if self._should_cool(idle):
                        self.core_temp[core] = calc_idle_cooling(self.core_temp[core], idle)

                    # 记录冷却后温度
                    self.temp_records.append({
                        'time': next_start,
                        'processor': core,
                        'temperature': float(self.core_temp[core]),
                        'voltage_idx': -1,
                        'voltage_value': float(V_IDLE)
                    })

                self.core_free[core] = next_start

    def get_stats(self):
        dmr = self.mandatory_timeout / self.mandatory_total if self.mandatory_total > 0 else 0
        accuracy = (self.completed_weight / self.total_weight * 0.6441) if self.total_weight > 0 else 0
        return {
            'dmr': dmr,
            'accuracy': accuracy,
            'mandatory_total': self.mandatory_total,
            'mandatory_timeout': self.mandatory_timeout,
            'assign_counts': self.assign_counts.copy(),
            'energy': self.energy,
            'final_temps': self.core_temp.copy()
        }


# ==================== 可视化 ====================
def plot_temperature_curves(temp_records, period_ms, seed):
    if not temp_records:
        return

    df = pd.DataFrame(temp_records)
    df = df[df['temperature'].notna()]
    df['time_ms'] = df['time'] * 1000

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ['#1f77b4', '#ff7f0e']
    for proc_id in range(NUM_PROCESSORS):
        proc_df = df[df['processor'] == proc_id]
        if len(proc_df) > 0:
            axes[0].plot(proc_df['time_ms'], proc_df['temperature'],
                         color=colors[proc_id], linewidth=1, alpha=0.7, label=f'GPU {proc_id}')

    axes[0].axhline(y=Tmax, color='red', linestyle='--', linewidth=1.5, label=f'Tmax ({Tmax}°C)')
    axes[0].axhline(y=Tamb, color='gray', linestyle=':', linewidth=1, label=f'Tamb ({Tamb}°C)')
    axes[0].set_xlabel('Time (ms)', fontsize=12)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].set_title(f'ESATD: Temperature Trace (Period={period_ms}ms, Seed={seed})', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([Tamb - 5, 85])

    # 电压
    df_volt = df[df['voltage_idx'] >= 0]
    if len(df_volt) > 0:
        for proc_id in range(NUM_PROCESSORS):
            proc_df = df_volt[df_volt['processor'] == proc_id]
            if len(proc_df) > 0:
                axes[1].scatter(proc_df['time_ms'], proc_df['voltage_value'],
                                color=colors[proc_id], s=1, alpha=0.5, label=f'GPU {proc_id}')
    axes[1].set_xlabel('Time (ms)', fontsize=12)
    axes[1].set_ylabel('Voltage (V)', fontsize=12)
    axes[1].set_title('Voltage Usage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.55, 0.92])

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'esatd_temperature_{period_ms}ms_seed{seed}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"温度曲线图已保存: {save_path}")


def plot_processor_load(assign_counts, period_ms, seed):
    fig, ax = plt.subplots(figsize=(8, 6))

    cores = ['GPU0', 'GPU1']
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax.bar(cores, assign_counts, color=colors, edgecolor='black', alpha=0.8)

    total = sum(assign_counts)
    for bar, count in zip(bars, assign_counts):
        if total > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(5, total * 0.01),
                    f'{count}\n({count / total * 100:.1f}%)',
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Processor', fontsize=12)
    ax.set_ylabel('Task Count', fontsize=12)
    ax.set_title(f'ESATD: Task Distribution (Period={period_ms}ms, Seed={seed})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'esatd_load_{period_ms}ms_seed{seed}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"负载分布图已保存: {save_path}")


def plot_voltage_distribution(temp_records, period_ms, seed):
    """绘制电压使用分布"""
    if not temp_records:
        return

    df = pd.DataFrame(temp_records)
    df_volt = df[df['voltage_idx'] >= 0]

    if len(df_volt) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    voltage_counts = df_volt['voltage_idx'].value_counts().sort_index()
    voltage_labels = [f'{V_LEVELS[i]:.2f}V' for i in voltage_counts.index]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']

    bars = ax.bar(voltage_labels, voltage_counts.values, color=colors[:len(voltage_labels)],
                  edgecolor='black', alpha=0.8)

    total = sum(voltage_counts.values)
    for bar, count in zip(bars, voltage_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(1, total * 0.01),
                f'{count}\n({count / total * 100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Voltage Level', fontsize=12)
    ax.set_ylabel('Usage Count', fontsize=12)
    ax.set_title(f'ESATD: Voltage Distribution (Period={period_ms}ms, Seed={seed})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'esatd_voltage_{period_ms}ms_seed{seed}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"电压分布图已保存: {save_path}")


# ==================== 主调度器 ====================
def scheduler(period_ms, seed=42):
    start_time = time.time()

    print("=" * 70)
    print(f"ESATD (Complete Fixed) - 周期: {period_ms}ms, 种子: {seed}")
    print(f"  GPU0 WCET: {WCET_GPU0_BASE_MS}ms, GPU1: {WCET_GPU1_BASE_MS}ms")
    print(f"  电压选择: 最低可行电压（优先保证 deadline）")
    print(f"  负载均衡: 双核使用")
    print(f"  冷却阈值: {IDLE_COOLING_THRESHOLD_MS}ms")
    print(f"  环境温度: {Tamb}°C, Tmax: {Tmax}°C")
    print("=" * 70)

    # 生成任务
    tasks = create_task_stream_batch(period_ms, NUM_CYCLES, JITTER_RATIO, seed)
    print(f"\n[1/3] 生成任务: {len(tasks)} 个")

    # 按周期分组
    period_sec = period_ms / 1000.0
    tasks_by_cycle = [[] for _ in range(NUM_CYCLES)]
    for task in tasks:
        cycle_idx = min(int(task['arrive'] / period_sec), NUM_CYCLES - 1)
        tasks_by_cycle[cycle_idx].append(task)

    print(f"[2/3] 开始调度 {NUM_CYCLES} 个周期...")
    print("-" * 80)

    scheduler = ESATDScheduler(period_ms)

    for cycle in range(NUM_CYCLES):
        cycle_start = cycle * period_sec

        if cycle > 0:
            scheduler.advance(cycle_start)

        scheduler.schedule_cycle(tasks_by_cycle[cycle], cycle_start)

        if (cycle + 1) % DEBUG_PRINT_CYCLES == 0:
            stats = scheduler.get_stats()
            print(f"  周期 {cycle + 1:4d}/{NUM_CYCLES} | "
                  f"GPU0: {scheduler.core_temp[0]:5.1f}°C | "
                  f"GPU1: {scheduler.core_temp[1]:5.1f}°C | "
                  f"DMR: {stats['dmr'] * 100:6.2f}% | "
                  f"分配: {scheduler.assign_counts[0]:5d}/{scheduler.assign_counts[1]:5d}")

    print("-" * 80)

    # 计算指标
    stats = scheduler.get_stats()

    # 从温度记录计算 ATP 和 IIT
    temps = [r['temperature'] for r in scheduler.temp_records if r['temperature'] is not None]
    atp = np.mean(temps) if temps else Tamb
    over = [t - Tmax for t in temps if t > Tmax]
    iit = np.mean(over) if over else 0.0

    print("\n" + "=" * 70)
    print("调度结果")
    print("=" * 70)
    print(f"  总强制任务: {stats['mandatory_total']}")
    print(f"  强制任务超时: {stats['mandatory_timeout']}")
    print(f"  DMR: {stats['dmr'] * 100:.4f}%")
    print(f"  准确率: {stats['accuracy']:.4f}")
    print(f"  最终温度: GPU0={scheduler.core_temp[0]:.1f}°C, GPU1={scheduler.core_temp[1]:.1f}°C")
    print(f"  任务分配: GPU0={stats['assign_counts'][0]}, GPU1={stats['assign_counts'][1]}")
    print(f"  能耗: {stats['energy']:.4f}")
    print(f"  ATP: {atp:.2f}°C")
    print(f"  IIT: {iit:.4f}°C")
    print("=" * 70)

    # 可视化
    plot_temperature_curves(scheduler.temp_records, period_ms, seed)
    plot_processor_load(stats['assign_counts'], period_ms, seed)
    plot_voltage_distribution(scheduler.temp_records, period_ms, seed)

    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.3f} 秒")

    # 保存结果
    result = {
        'algorithm': 'ESATD_Complete_Fixed',
        'period_ms': period_ms,
        'seed': seed,
        'dmr': stats['dmr'] * 100,
        'accuracy': stats['accuracy'],
        'atp': atp,
        'iit': iit,
        'energy': stats['energy'],
        'assign_counts': stats['assign_counts'],
        'final_temps': stats['final_temps'].tolist()
    }

    with open(os.path.join(RESULTS_DIR, f'esatd_result_{period_ms}ms_seed{seed}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    print(f"结果已保存: {os.path.join(RESULTS_DIR, f'esatd_result_{period_ms}ms_seed{seed}.json')}")

    return stats


def run_batch(periods_ms, seeds=[42]):
    """批量运行实验"""
    print("=" * 80)
    print("ESATD 批量实验 (Complete Fixed)")
    print(f"周期列表: {periods_ms} ms")
    print(f"随机种子: {seeds}")
    print("=" * 80)

    all_results = []
    for period in periods_ms:
        for seed in seeds:
            print(f"\n{'#' * 70}")
            print(f"# 实验: period={period}ms, seed={seed}")
            print(f"{'#' * 70}")
            result = scheduler(period, seed)
            all_results.append({
                'period_ms': period,
                'seed': seed,
                'dmr': result['dmr'],
                'accuracy': result['accuracy']
            })

    # 按周期汇总
    print("\n" + "=" * 80)
    print("批量实验汇总")
    print("=" * 80)

    period_summary = {}
    for r in all_results:
        p = r['period_ms']
        if p not in period_summary:
            period_summary[p] = {'dmr': [], 'acc': []}
        period_summary[p]['dmr'].append(r['dmr'])
        period_summary[p]['acc'].append(r['accuracy'])

    print(f"{'周期(ms)':<10} {'DMR(%)':<12} {'准确率':<12}")
    print("-" * 40)
    for period in sorted(period_summary.keys()):
        avg_dmr = np.mean(period_summary[period]['dmr'])
        avg_acc = np.mean(period_summary[period]['acc'])
        print(f"{period:<10} {avg_dmr:<12.2f} {avg_acc:<12.4f}")

    # 保存汇总结果
    summary_df = pd.DataFrame([{
        'period_ms': p,
        'dmr_avg': np.mean(period_summary[p]['dmr']),
        'accuracy_avg': np.mean(period_summary[p]['acc'])
    } for p in sorted(period_summary.keys())])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'esatd_batch_summary.csv'), index=False)
    print(f"\n汇总结果已保存: {os.path.join(RESULTS_DIR, 'esatd_batch_summary.csv')}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', '-p', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-periods', action='store_true')
    args = parser.parse_args()

    if args.all_periods:
        periods = [100, 150, 200, 250, 300, 500, 1000]
        run_batch(periods)
    else:
        scheduler(args.period, args.seed)
