"""
Phase 4: Policy Buffer — 이중 슬롯, Atomic Swap, 보간·히스테리시스
논문: Active/Standby, 급격한 전환 완화, Ping-pong 방지.
"""
from __future__ import annotations

import threading
from typing import Tuple, Optional
import numpy as np


class PolicyBuffer:
    """이중 버퍼: Active(현재 γ, β) / Standby(새 γ', β'). Atomic Swap. 보간·히스테리시스."""

    def __init__(
        self,
        clip: float = 5.0,
        interpolate: bool = True,
        interpolation_steps: int = 5,
        hysteresis_forward: float = 0.3,
        hysteresis_back: float = 0.5,
    ):
        self.clip = clip
        self.interpolate = interpolate
        self.interpolation_steps = max(1, interpolation_steps)
        self.hysteresis_forward = hysteresis_forward
        self.hysteresis_back = hysteresis_back
        self._active = {"gamma": np.array(1.0, dtype=np.float32), "beta": np.array(0.0, dtype=np.float32)}
        self._standby = {"gamma": np.array(1.0, dtype=np.float32), "beta": np.array(0.0, dtype=np.float32)}
        self._lock = threading.Lock()
        self._last_written = {"gamma": np.array(1.0, dtype=np.float32), "beta": np.array(0.0, dtype=np.float32)}
        self._step = 0

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """System 1: Non-blocking Read. 반환 (γ, β). 보간 중이면 이전·신규 블렌드."""
        with self._lock:
            return self._active["gamma"].copy(), self._active["beta"].copy()

    def write(self, gamma: np.ndarray, beta: np.ndarray) -> None:
        """System 2: Standby에 기록. 히스테리시스 적용 후 Atomic Swap. 보간 옵션."""
        g = np.clip(np.asarray(gamma, dtype=np.float32).ravel(), -self.clip, self.clip)
        b = np.clip(np.asarray(beta, dtype=np.float32).ravel(), -self.clip, self.clip)
        with self._lock:
            cur_g = self._active["gamma"].copy()
            cur_b = self._active["beta"].copy()
            diff_g = np.abs(g - cur_g).max()
            diff_b = np.abs(b - cur_b).max()
            thresh_forward = self.hysteresis_forward
            thresh_back = self.hysteresis_back
            if diff_g < thresh_forward and diff_b < thresh_forward:
                return
            self._standby["gamma"] = g
            self._standby["beta"] = b
            if self.interpolate and self.interpolation_steps > 1:
                self._interpolation_step = 0
                self._interp_from_g = cur_g.copy()
                self._interp_from_b = cur_b.copy()
                self._interp_to_g = g.copy()
                self._interp_to_b = b.copy()
            self._active, self._standby = self._standby, self._active
            if self.interpolate and self.interpolation_steps > 1:
                self._active["gamma"] = cur_g.copy()
                self._active["beta"] = cur_b.copy()
            self._last_written["gamma"] = g
            self._last_written["beta"] = b

    def tick_interpolate(self) -> None:
        """보간: 매 호출 시 한 스텝씩 active를 from→to 쪽으로 이동. (Fast Loop에서 호출 가능)"""
        with self._lock:
            step = getattr(self, "_interpolation_step", self.interpolation_steps)
            if step >= self.interpolation_steps:
                return
            step += 1
            self._interpolation_step = step
            alpha = step / self.interpolation_steps
            from_g = getattr(self, "_interp_from_g", self._active["gamma"]).copy()
            from_b = getattr(self, "_interp_from_b", self._active["beta"]).copy()
            to_g = getattr(self, "_interp_to_g", self._active["gamma"]).copy()
            to_b = getattr(self, "_interp_to_b", self._active["beta"]).copy()
            self._active["gamma"] = ((1 - alpha) * from_g + alpha * to_g).astype(np.float32)
            self._active["beta"] = ((1 - alpha) * from_b + alpha * to_b).astype(np.float32)
