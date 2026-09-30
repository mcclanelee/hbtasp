"""Analytic first-order RC thermal intervals and runtime disturbances."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class ThermalParameters:
    a: float = 0.1
    alpha: float = 41.01
    gamma: float = 512.635
    B: float = 0.5058
    tmax: float = 60.0

    def steady_state(self, voltage: float) -> float:
        return self.a * (self.alpha + self.gamma * voltage**3) / self.B

    def transition(self, start_temp: float, voltage: float, duration_s: float) -> float:
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        tss = self.steady_state(voltage)
        return tss + (start_temp - tss) * math.exp(-self.B * duration_s)

    def threshold_crossing(self, start_temp: float, voltage: float) -> float | None:
        """Return non-negative crossing time, or None when no future crossing exists."""
        tss = self.steady_state(voltage)
        if math.isclose(start_temp, self.tmax, abs_tol=1e-14):
            return 0.0
        denominator = start_temp - tss
        numerator = self.tmax - tss
        if denominator == 0 or numerator / denominator <= 0:
            return None
        crossing = -math.log(numerator / denominator) / self.B
        return crossing if crossing >= 0 else None

    def interval_metrics(self, start_temp: float, voltage: float, duration_s: float) -> dict:
        end_temp = self.transition(start_temp, voltage, duration_s)
        tss = self.steady_state(voltage)
        peak = max(start_temp, end_temp)
        if duration_s == 0 or peak <= self.tmax:
            return {"end_temp": end_temp, "peak_temp": peak,
                    "iit_Cs": 0.0, "violation_duration_s": 0.0}

        crossing = self.threshold_crossing(start_temp, voltage)
        if start_temp > self.tmax:
            above_start = 0.0
            above_end = duration_s if crossing is None else min(duration_s, crossing)
        else:
            if crossing is None or crossing >= duration_s:
                return {"end_temp": end_temp, "peak_temp": peak,
                        "iit_Cs": 0.0, "violation_duration_s": 0.0}
            above_start, above_end = crossing, duration_s

        def primitive_delta(x: float, y: float) -> float:
            return ((tss - self.tmax) * (y - x)
                    + (start_temp - tss) / self.B
                    * (math.exp(-self.B * x) - math.exp(-self.B * y)))

        iit = max(0.0, primitive_delta(above_start, above_end))
        return {"end_temp": end_temp, "peak_temp": peak, "iit_Cs": iit,
                "violation_duration_s": max(0.0, above_end - above_start)}

    def critical_overrun(self, predicted_end_temp: float, voltage: float) -> float:
        """Minimum extra active time that makes the end temperature exceed Tmax."""
        if predicted_end_temp >= self.tmax:
            return 0.0
        tss = self.steady_state(voltage)
        if tss <= self.tmax:
            return math.inf
        headroom = self.tmax - predicted_end_temp
        heating_gap = tss - predicted_end_temp
        if headroom >= heating_gap:
            return math.inf
        return -math.log(1.0 - headroom / heating_gap) / self.B


@dataclass(frozen=True)
class DisturbanceParameters:
    dispatch_probability: float = 0.0
    dispatch_max_s: float = 0.0
    execution_probability: float = 0.0
    execution_overrun_max_fraction: float = 0.0

    def validate(self) -> None:
        for value in (self.dispatch_probability, self.execution_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("probabilities must be in [0, 1]")
        if self.dispatch_max_s < 0 or self.execution_overrun_max_fraction < 0:
            raise ValueError("disturbance magnitudes must be non-negative")


class RuntimeDisturbance:
    def __init__(self, params: DisturbanceParameters, seed: int):
        params.validate()
        self.params = params
        self.rng = random.Random(seed)

    def realize(self, planned_exec_s: float) -> tuple[float, float]:
        dispatch = 0.0
        if self.rng.random() < self.params.dispatch_probability:
            dispatch = self.rng.uniform(0.0, self.params.dispatch_max_s)
        overrun = 0.0
        if self.rng.random() < self.params.execution_probability:
            overrun = planned_exec_s * self.rng.uniform(
                0.0, self.params.execution_overrun_max_fraction
            )
        return dispatch, overrun
