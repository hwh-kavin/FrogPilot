import time
from enum import Enum, auto

from openpilot.common.params import Params


PARAM_REFRESH_SEC = 2.0


class HTDState(Enum):
  INACTIVE = auto()
  PAUSED = auto()
  WAITING_RESUME = auto()


class HumanTurnDetection:
  def __init__(self) -> None:
    self._params = Params()
    self._last_params_read = 0.0

    self._enabled = False
    self._trigger_angle_threshold_deg = 45.0
    self._resume_angle_diff_threshold_deg = 10.0
    self._resume_delay_sec = 0.3

    self._state: HTDState = HTDState.INACTIVE
    self._resume_condition_start_time = 0.0

  def _read_params(self) -> None:
    now = time.monotonic()
    if now - self._last_params_read < PARAM_REFRESH_SEC:
      return
    self._last_params_read = now

    self._enabled = self._params.get_bool("dp_htd_enabled")
    self._trigger_angle_threshold_deg = self._clamp_int("dp_htd_turn_angle_threshold", 45, 40, 90)
    self._resume_angle_diff_threshold_deg = self._clamp_int("dp_htd_resume_angle_diff_threshold", 10, 5, 20)
    resume_delay_ms = self._clamp_int("dp_htd_resume_delay_ms", 300, 0, 5000)
    self._resume_delay_sec = resume_delay_ms / 1000.0

  def update(
    self,
    lat_active: bool,
    steering_angle_deg: float,
    steering_pressed: bool,
    model_expected_angle_deg: float,
  ) -> tuple[bool, HTDState]:
    self._read_params()

    # 恢复1：总开关关闭，立即恢复
    if not self._enabled:
      self._state = HTDState.INACTIVE
      self._resume_condition_start_time = 0.0
      return True, self._state

    if not lat_active:
      self._state = HTDState.INACTIVE
      self._resume_condition_start_time = 0.0
      return True, self._state

    steering_angle_abs = abs(steering_angle_deg)

    # 触发：开关打开 + 手握方向盘 + 角度阈值
    if self._state == HTDState.INACTIVE:
      if steering_pressed and steering_angle_abs >= self._trigger_angle_threshold_deg:
        self._state = HTDState.PAUSED
        self._resume_condition_start_time = 0.0
        return False, self._state
      return True, self._state

    # 恢复2：模型期望角度与方向盘角度差值足够小
    angle_diff = abs(model_expected_angle_deg - steering_angle_deg)
    angle_aligned = angle_diff < self._resume_angle_diff_threshold_deg
    # 恢复3：没有手握方向盘
    hands_off = not steering_pressed
    resume_condition = angle_aligned or hands_off

    if resume_condition:
      if self._state != HTDState.WAITING_RESUME:
        self._state = HTDState.WAITING_RESUME
        self._resume_condition_start_time = time.monotonic()

      if time.monotonic() - self._resume_condition_start_time >= self._resume_delay_sec:
        self._state = HTDState.INACTIVE
        self._resume_condition_start_time = 0.0
        return True, self._state
    else:
      self._state = HTDState.PAUSED
      self._resume_condition_start_time = 0.0

    return False, self._state

  def _clamp_int(self, key: str, default: int, min_value: int, max_value: int) -> int:
    try:
      val = self._params.get(key)
      if val is None:
        return default
      return max(min(int(val), max_value), min_value)
    except Exception:
      return default
