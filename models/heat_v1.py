"""
HEAT Algorithm - Paper Compliant Version
按照论文 Algorithm 1-5 完整实现

核心特点：
1. 时间片划分 (Deadline Partitioning)
2. 聚类设计 (CLUSTER-DESIGN) - 将核心配对
3. 调度设计 (SCHEDULE-DESIGN) - 任务分配
4. 温度设计 (TEMP-DESIGN) - 冷热交替 + 温度感知
5. 能量设计 (ENERGY-DESIGN) - 公式(4)电压选择
"""

import numpy as np
import math
import pandas as pd
import heapq
import os
import json
from datetime import datetime
import argparse

# ==================== 参数配置 ====================
NUM_PROCESSORS = 2
NUM_TASKS_PER_CYCLE = 4
SUBTASKS_PER_TASK = 4

# DNN 参数 - 固定 Level 2
FIXED_DNN_LEVEL = 2
WCET_GPU0_BASE_MS = [6.3, 9.0, 12.3, 14.8, 20.1][FIXED_DNN_LEVEL]
WCET_GPU1_BASE_MS = [13.0, 18.2, 25.1, 29.5, 40.0][FIXED_DNN_LEVEL]
FIXED_ACCURACY = [0.5790, 0.6183, 0.6441, 0.6730, 0.6884][FIXED_DNN_LEVEL]

# DVFS 参数
V_REF = 0.8
V_LEVELS = {0: 0.6, 1: 0.7, 2: 0.8, 3: 0.85, 4: 0.88}
DEFAULT_V_IDX = 2
MIN_VOLTAGE_IDX = 0
MAX_VOLTAGE_IDX = 4

# 温度参数
a = 0.1
alpha = 41.01
gamma = 512.635
B = 0.5058
Tmax = 60.0
Tamb = 25.0
T_HOT = 55.0
V_IDLE = 0.4

# 调度参数
NUM_CYCLES = 1000
JITTER_RATIO = 0.1
EARLY_EXIT_THRESHOLD = 0.65
LOW_WEIGHT_THRESHOLD = 0.3


# ==================== 温度模型 ====================
def calc_steady_state_temp(v_val):
    A = a * (alpha + gamma * math.pow(v_val, 3))
    return A / B


def calc_temperature_change(T0, v_val, time_sec):
    if time_sec <= 1e-6:
        return T0
    Tss = calc_steady_state_temp(v_val)
    exp_term = math.exp(-B * time_sec)
    Te = Tss - (Tss - T0) * exp_term
    return max(Tamb, min(Te, 120.0))


def get_wcet_sec(core_id, v_idx):
    base_ms = WCET_GPU0_BASE_MS if core_id == 0 else WCET_GPU1_BASE_MS
    v_target = V_LEVELS[v_idx]
    return (base_ms * (V_REF / v_target)) / 1000.0


# ==================== 任务类 ====================
class Subtask:
    def __init__(self, idx, weight, is_mandatory=False):
        self.idx = idx
        self.weight = weight
        self.is_mandatory = is_mandatory
        self.task = None


class Task:
    def __init__(self, task_id, jobid, deadline_sec, arrive_time):
        self.id = task_id
        self.jobid = jobid
        self.arrive_time = arrive_time
        self.deadline_sec = deadline_sec
        self.abs_deadline = arrive_time + deadline_sec
        self.subtasks = []
        self.mandatory_subtask = None
        self.optional_subtasks = []


class SchedulableSubtask:
    def __init__(self, subtask, task, task_idx, release_time):
        self.subtask = subtask
        self.task = task
        self.task_idx = task_idx
        self.subtask_idx = subtask.idx
        self.abs_deadline = task.abs_deadline
        self.weight = subtask.weight
        self.is_mandatory = subtask.is_mandatory
        self.accuracy = FIXED_ACCURACY
        self.steady_state_temp = None
        self.assigned_core = None
        self.finish_time = None
        self.is_timeout = False
        self.is_migrating = False  # 是否是迁移任务

    def get_wcet_sec(self, v_idx):
        return get_wcet_sec(self.assigned_core, v_idx)


# ==================== 任务生成 ====================
def generate_task_weights(num_subtasks=4, seed=None):
    if seed is not None:
        np.random.seed(seed)
    weights = []
    for _ in range(num_subtasks):
        if np.random.random() < 0.25:
            weights.append(np.random.uniform(0.4, 0.7))
        else:
            weights.append(np.random.uniform(0.05, 0.2))
    total = sum(weights)
    return [w / total for w in weights]


def create_task_stream(period_ms, num_cycles, jitter_ratio=0.1, seed=42):
    np.random.seed(seed)
    period_sec = period_ms / 1000.0
    jitter_bound = jitter_ratio * period_sec

    task_stream = []
    job_counter = {i: 0 for i in range(NUM_TASKS_PER_CYCLE)}
    task_weights = [generate_task_weights(SUBTASKS_PER_TASK, seed + i)
                    for i in range(NUM_TASKS_PER_CYCLE)]

    for task_id in range(NUM_TASKS_PER_CYCLE):
        phase = (task_id / NUM_TASKS_PER_CYCLE) * period_sec
        weights = task_weights[task_id]

        for k in range(num_cycles):
            jitter = np.random.uniform(-jitter_bound, jitter_bound)
            arrive_time = max(0, phase + k * period_sec + jitter)
            job_counter[task_id] += 1

            task = Task(task_id, job_counter[task_id], period_sec, arrive_time)

            for sub_idx, weight in enumerate(weights):
                is_mandatory = (weight == max(weights))
                subtask = Subtask(sub_idx, weight, is_mandatory)
                subtask.task = task
                task.subtasks.append(subtask)
                if is_mandatory:
                    task.mandatory_subtask = subtask
                else:
                    task.optional_subtasks.append(subtask)

            task_stream.append(task)

    task_stream.sort(key=lambda x: x.arrive_time)
    return task_stream


# ==================== HEAT 论文算法实现 ====================

def cluster_design(subtasks, processors):
    """
    Algorithm 2: CLUSTER-DESIGN
    将核心配对成双核集群，每个集群最多两个核心
    """
    m = len(processors)
    clusters = []
    task_cluster_assign = {}
    core_cluster_assign = {}

    # 按最低 share 值排序
    share_list = []
    for sst in subtasks:
        for core in range(m):
            wcet = get_wcet_sec(core, DEFAULT_V_IDX)
            share_list.append((wcet, sst, core))

    share_list.sort(key=lambda x: x[0])

    for wcet, sst, core in share_list:
        if sst in task_cluster_assign:
            continue

        if core not in core_cluster_assign:
            # 找第二个最优核心
            second_best_core = None
            second_best_wcet = float('inf')
            for other_core in range(m):
                if other_core != core and other_core not in core_cluster_assign:
                    w = get_wcet_sec(other_core, DEFAULT_V_IDX)
                    if w < second_best_wcet:
                        second_best_wcet = w
                        second_best_core = other_core

            if second_best_core is not None:
                # 创建新集群
                cluster_id = len(clusters)
                clusters.append({
                    'id': cluster_id,
                    'cores': [core, second_best_core],
                    'tasks': [sst]
                })
                core_cluster_assign[core] = cluster_id
                core_cluster_assign[second_best_core] = cluster_id
                task_cluster_assign[sst] = cluster_id
            else:
                # 单核集群
                cluster_id = len(clusters)
                clusters.append({
                    'id': cluster_id,
                    'cores': [core],
                    'tasks': [sst]
                })
                core_cluster_assign[core] = cluster_id
                task_cluster_assign[sst] = cluster_id
        else:
            # 核心已有集群，尝试添加任务
            cluster_id = core_cluster_assign[core]
            clusters[cluster_id]['tasks'].append(sst)
            task_cluster_assign[sst] = cluster_id

    return clusters


def schedule_design(cluster, period_sec):
    """
    Algorithm 3: SCHEDULE-DESIGN
    在集群内的核心上分配任务
    """
    cores = cluster['cores']
    tasks = cluster['tasks']

    if len(cores) == 1:
        # 单核：所有任务都分配到这个核心
        for sst in tasks:
            sst.assigned_core = cores[0]
            sst.is_migrating = False
        return

    # 双核：根据任务偏好分配
    core0, core1 = cores[0], cores[1]

    # 计算每个任务在两个核心上的利用率比
    task_ratios = []
    for sst in tasks:
        w0 = get_wcet_sec(core0, DEFAULT_V_IDX)
        w1 = get_wcet_sec(core1, DEFAULT_V_IDX)
        ratio = w0 / w1 if w1 > 0 else float('inf')
        task_ratios.append((ratio, sst))

    task_ratios.sort(key=lambda x: x[0])

    # 按比例分配
    spare_core0 = period_sec
    spare_core1 = period_sec

    # 从前往后分配给 core0（偏好 core0 的任务）
    i = 0
    while i < len(task_ratios) and task_ratios[i][0] <= 1 and spare_core0 > 0:
        ratio, sst = task_ratios[i]
        wcet = get_wcet_sec(core0, DEFAULT_V_IDX)
        if wcet <= spare_core0 + 1e-6:
            sst.assigned_core = core0
            sst.is_migrating = False
            spare_core0 -= wcet
        i += 1

    # 从后往前分配给 core1（偏好 core1 的任务）
    j = len(task_ratios) - 1
    while j >= i and spare_core1 > 0:
        ratio, sst = task_ratios[j]
        wcet = get_wcet_sec(core1, DEFAULT_V_IDX)
        if wcet <= spare_core1 + 1e-6:
            sst.assigned_core = core1
            sst.is_migrating = False
            spare_core1 -= wcet
        j -= 1

    # 剩余任务分配给空闲容量更多的核心
    remaining = task_ratios[i:j + 1] if i <= j else []
    for ratio, sst in remaining:
        if spare_core0 >= spare_core1:
            sst.assigned_core = core0
            spare_core0 -= get_wcet_sec(core0, DEFAULT_V_IDX)
        else:
            sst.assigned_core = core1
            spare_core1 -= get_wcet_sec(core1, DEFAULT_V_IDX)
        sst.is_migrating = False


def temp_design(subtasks, period_sec):
    """
    Algorithm 4: TEMP-DESIGN
    温度感知调度：冷热任务交替
    """
    # 按核心分组
    core_tasks = {}
    for sst in subtasks:
        if sst.assigned_core not in core_tasks:
            core_tasks[sst.assigned_core] = []
        core_tasks[sst.assigned_core].append(sst)

    scheduled_order = []

    for core, tasks in core_tasks.items():
        # 计算每个任务的稳态温度（在分配的电压下）
        for sst in tasks:
            v_idx = DEFAULT_V_IDX  # 先用默认电压估算
            v_val = V_LEVELS[v_idx]
            sst.steady_state_temp = calc_steady_state_temp(v_val)

        # 计算平均稳态温度
        avg_temp = np.mean([s.steady_state_temp for s in tasks]) if tasks else Tamb

        # 分类 hot/cold
        hot = [s for s in tasks if s.steady_state_temp >= avg_temp]
        cold = [s for s in tasks if s.steady_state_temp < avg_temp]

        # 排序
        hot.sort(key=lambda s: s.steady_state_temp, reverse=True)
        cold.sort(key=lambda s: s.steady_state_temp)

        # 交替排列
        interleaved = []
        i, j = 0, 0
        while i < len(hot) or j < len(cold):
            if i < len(hot):
                interleaved.append(hot[i])
                i += 1
            if j < len(cold):
                interleaved.append(cold[j])
                j += 1

        scheduled_order.extend(interleaved)

    return scheduled_order


def energy_design(core_tasks, core_migrating_tasks, period_sec, current_temp):
    """
    Algorithm 5: ENERGY-DESIGN
    按照论文公式 (4) 选择电压

    F_opt = Σ(fixed_tasks) / (|S_k| - Σ(migrating_tasks))
    """
    # 计算固定任务总执行时间
    total_fixed_time = 0.0
    for sst in core_tasks:
        if not sst.is_migrating:
            total_fixed_time += get_wcet_sec(sst.assigned_core, DEFAULT_V_IDX)

    # 计算迁移任务总执行时间
    total_migrating_time = 0.0
    for sst in core_migrating_tasks:
        total_migrating_time += get_wcet_sec(sst.assigned_core, DEFAULT_V_IDX)

    # 公式 (4)
    denominator = period_sec - total_migrating_time
    if denominator <= 1e-6:
        required_freq = 1.0
    else:
        required_freq = total_fixed_time / denominator

    # 限制频率范围
    required_freq = max(0.3, min(1.0, required_freq))

    # 频率转电压（假设线性关系）
    v_max = V_LEVELS[MAX_VOLTAGE_IDX]
    v_min = V_LEVELS[MIN_VOLTAGE_IDX]
    target_voltage = v_min + required_freq * (v_max - v_min)

    # 选择最接近的电压档位
    best_v_idx = DEFAULT_V_IDX
    best_diff = float('inf')
    for v_idx, v_val in V_LEVELS.items():
        diff = abs(v_val - target_voltage)
        if diff < best_diff:
            best_diff = diff
            best_v_idx = v_idx

    # 温度补偿：过热时升压
    if current_temp > T_HOT:
        best_v_idx = min(MAX_VOLTAGE_IDX, best_v_idx + 1)
    elif current_temp < Tamb + 5:
        best_v_idx = max(MIN_VOLTAGE_IDX, best_v_idx - 1)

    return best_v_idx


# ==================== 主调度器 ====================
class HEAT_PaperScheduler:
    def __init__(self, period_ms):
        self.period_sec = period_ms / 1000.0
        self.core_next_free = [0.0, 0.0]
        self.core_temps = [Tamb, Tamb]
        self.core_voltages = [DEFAULT_V_IDX, DEFAULT_V_IDX]
        self.defect_detected = set()

        # 统计
        self.mandatory_total = 0
        self.mandatory_timeout = 0
        self.mandatory_completed = 0
        self.processor_assign_count = [0, 0]

        # 温度记录
        self.temperature_records = []
        self.voltage_records = []

    def record_temperature(self, proc_id, time, temp, v_idx):
        self.temperature_records.append({
            'time': time, 'processor': proc_id,
            'temperature': temp, 'voltage_idx': v_idx,
            'voltage_value': V_LEVELS.get(v_idx, V_IDLE)
        })

    def record_voltage(self, proc_id, time, v_idx):
        self.voltage_records.append({
            'time': time, 'processor': proc_id,
            'voltage_idx': v_idx, 'voltage_value': V_LEVELS[v_idx]
        })

    def simulate_execution(self, core, sst, start_time, start_temp, v_idx):
        wcet = sst.get_wcet_sec(v_idx)
        finish_time = start_time + wcet
        is_timeout = finish_time > sst.abs_deadline + 1e-3

        # 温度变化
        v_val = V_LEVELS[v_idx]
        end_temp = calc_temperature_change(start_temp, v_val, wcet)

        # 记录
        self.record_temperature(core, start_time, start_temp, v_idx)
        self.record_temperature(core, finish_time, end_temp, v_idx)

        return finish_time, end_temp, is_timeout

    def schedule_cycle(self, cycle_tasks, cycle_start_time):
        """按照 HEAT 论文 Algorithm 1 调度一个周期"""

        # ===== 步骤1: 时间片内的任务准备 =====
        subtasks = []
        for task in cycle_tasks:
            for subtask in task.subtasks:
                sst = SchedulableSubtask(subtask, task, task.id, cycle_start_time)
                subtasks.append(sst)
                if sst.is_mandatory:
                    self.mandatory_total += 1

        if not subtasks:
            return

        # ===== 步骤2: CLUSTER-DESIGN (Algorithm 2) =====
        processors = list(range(NUM_PROCESSORS))
        clusters = cluster_design(subtasks, processors)

        # ===== 步骤3: SCHEDULE-DESIGN (Algorithm 3) =====
        for cluster in clusters:
            schedule_design(cluster, self.period_sec)

        # ===== 步骤4: TEMP-DESIGN (Algorithm 4) =====
        scheduled_order = temp_design(subtasks, self.period_sec)

        # ===== 步骤5: ENERGY-DESIGN (Algorithm 5) + 执行 =====
        # 按核心分组
        core_subtasks = {0: [], 1: []}
        for sst in scheduled_order:
            if sst.assigned_core is not None:
                core_subtasks[sst.assigned_core].append(sst)

        # 记录电压变化
        for core in range(NUM_PROCESSORS):
            self.record_voltage(core, cycle_start_time, self.core_voltages[core])

        # 执行
        for core in range(NUM_PROCESSORS):
            tasks = core_subtasks[core]
            if not tasks:
                continue

            # 计算迁移任务（分配给其他核心的任务）
            core_migrating = [s for s in tasks if s.is_migrating]

            # ENERGY-DESIGN: 选择电压
            v_idx = energy_design(tasks, core_migrating, self.period_sec, self.core_temps[core])

            if v_idx != self.core_voltages[core]:
                self.core_voltages[core] = v_idx
                self.record_voltage(core, cycle_start_time, v_idx)

            start_time = max(cycle_start_time, self.core_next_free[core])
            current_temp = self.core_temps[core]

            for sst in tasks:
                task_key = (sst.task.id, sst.task.jobid)

                # 早退检查
                if not sst.is_mandatory and task_key in self.defect_detected:
                    if sst.weight <= LOW_WEIGHT_THRESHOLD:
                        continue

                # 执行
                finish_time, end_temp, is_timeout = self.simulate_execution(
                    core, sst, start_time, current_temp, v_idx
                )

                # 更新状态
                current_temp = end_temp
                start_time = finish_time
                self.core_next_free[core] = finish_time
                self.processor_assign_count[core] += 1

                # 统计
                if sst.is_mandatory:
                    if is_timeout:
                        self.mandatory_timeout += 1
                    else:
                        self.mandatory_completed += 1

                    if sst.weight > EARLY_EXIT_THRESHOLD:
                        self.defect_detected.add(task_key)

            self.core_temps[core] = current_temp

        # 推进时间
        self.global_time = min(self.core_next_free)

    def advance_to_next_cycle(self, next_start):
        for core in range(NUM_PROCESSORS):
            if self.core_next_free[core] < next_start:
                idle_time = next_start - self.core_next_free[core]
                if idle_time > 0.001:
                    self.core_temps[core] = calc_temperature_change(
                        self.core_temps[core], V_IDLE, idle_time
                    )
                self.core_next_free[core] = next_start

    def get_stats(self):
        dmr = self.mandatory_timeout / self.mandatory_total if self.mandatory_total > 0 else 0
        return {
            'mandatory_total': self.mandatory_total,
            'mandatory_timeout': self.mandatory_timeout,
            'dmr': dmr,
            'processor_temps': self.core_temps,
            'processor_assign_count': self.processor_assign_count
        }


# ==================== 主函数 ====================
def scheduler(period_ms, seed=42):
    print("=" * 60)
    print(f"HEAT (Paper Compliant) - 周期: {period_ms}ms")
    print(f"  固定 DNN 等级: Level {FIXED_DNN_LEVEL}")
    print(f"  GPU0 WCET: {WCET_GPU0_BASE_MS}ms @0.8V")
    print(f"  GPU1 WCET: {WCET_GPU1_BASE_MS}ms @0.8V")
    print("=" * 60)

    # 生成任务
    task_stream = create_task_stream(period_ms, NUM_CYCLES, JITTER_RATIO, seed)
    print(f"  生成任务数: {len(task_stream)}")

    # 调度器
    scheduler = HEAT_PaperScheduler(period_ms)

    # 按周期调度
    for cycle in range(NUM_CYCLES):
        cycle_start = cycle * period_ms / 1000.0
        cycle_end = (cycle + 1) * period_ms / 1000.0

        # 获取当前周期任务
        cycle_tasks = [t for t in task_stream
                       if cycle_start <= t.arrive_time < cycle_end]

        if cycle > 0:
            scheduler.advance_to_next_cycle(cycle_start)

        if cycle_tasks:
            scheduler.schedule_cycle(cycle_tasks, cycle_start)

        if (cycle + 1) % 100 == 0:
            stats = scheduler.get_stats()
            print(f"[周期 {cycle + 1}] GPU0={scheduler.core_temps[0]:.1f}°C, "
                  f"GPU1={scheduler.core_temps[1]:.1f}°C, DMR={stats['dmr'] * 100:.2f}%")

    stats = scheduler.get_stats()
    print("\n" + "=" * 60)
    print(f"HEAT (Paper Compliant) 完成")
    print(f"  DMR: {stats['dmr'] * 100:.4f}%")
    print(f"  强制任务: {stats['mandatory_total']}, 超时: {stats['mandatory_timeout']}")
    print(f"  最终温度: GPU0={scheduler.core_temps[0]:.1f}°C, GPU1={scheduler.core_temps[1]:.1f}°C")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', '-p', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scheduler(args.period, args.seed)
