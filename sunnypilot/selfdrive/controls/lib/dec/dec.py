"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

AEM (Anti-Ghost Braking) ported into SP DynamicExperimentalController
"""

from cereal import messaging
from opendbc.car import structs
import numpy as np
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from typing import Literal

TRAJECTORY_SIZE = 33
ModeType = Literal['acc', 'blended']

class DECConfig:
  # --- 速度定義 ---
  HIGHWAY_SPEED_ON  = 70.0
  HIGHWAY_SPEED_OFF = 65.0

  # --- 靈敏度曲線 (KPH) ---
  SENSITIVITY_BP   = [0.,  50., 80., 110.]
  SENSITIVITY_VALS = [1.0, 1.0, 0.85, 0.4]

  # --- 減速模型 (M/S 對應 距離) ---
  SLOW_DOWN_BP   = [0.,  5.,   10.,  15.,  20.,  25.,   30.]
  SLOW_DOWN_DIST = [5.,  25.,  50.,  75.,  100., 130.,  160.]

  MODE_ACC = 'acc'
  MODE_BLENDED = 'blended'
  
  # --- 穩定前車檢測參數 ---
  LEAD_STABLE_FRAMES = 20           # 對應20幀（20Hz）= 1秒
  LEAD_STABLE_DIST_VAR = 3.0        
  LEAD_MIN_DIST = 2.0               
  
  # --- 實驗模式退出機制 ---
  EXIT_TRIGGER_THRESHOLD = 0.45     
  EXIT_DEBOUNCE_FRAMES = 5          


class SmoothKalmanFilter:
  """AEM 簡化版濾波器，專注於平滑運算與秒起步"""
  def __init__(self, initial_value=0.0):
    self.x = initial_value
    self.P = 1.0
    self.R = 0.2
    self.Q = 0.01
    self.initialized = False

  def add_data(self, measurement):
    if not self.initialized:
      self.x = measurement
      self.initialized = True
      return

    self.P = self.P + self.Q
    K = self.P / (self.P + self.R)
    
    # AEM 混合平滑因子
    smoothing_factor = 0.85
    effective_K = K * (1.0 - smoothing_factor) + smoothing_factor * 0.1
    
    self.x = self.x + effective_K * (measurement - self.x)
    self.P = (1 - effective_K) * self.P

  def get_value(self):
    return self.x if self.initialized else 0.0


class ModeTransitionManager:
  """AEM 模式切換管理器 (具備綠燈恢復與遲滯退出機制)"""
  def __init__(self):
    self.current_mode: ModeType = DECConfig.MODE_ACC
    self.mode_confidence = {DECConfig.MODE_ACC: 1.0, DECConfig.MODE_BLENDED: 0.0}
    self.low_urgency_counter = 0

  def request_mode(self, mode: ModeType, confidence: float = 1.0):
    # 綠燈快速恢復邏輯
    step = 0.2 if (mode == DECConfig.MODE_ACC and confidence >= 0.9) else 0.05

    target_conf = min(1.0, self.mode_confidence[mode] + step * confidence)
    self.mode_confidence[mode] = target_conf

    for m in self.mode_confidence:
      if m != mode:
        self.mode_confidence[m] = max(0.0, self.mode_confidence[m] - step)

    # 切換門檻
    threshold = 0.75 if mode != self.current_mode else 0.4
    if self.mode_confidence[mode] > threshold:
      self.current_mode = mode

  def update(self, urgency_val: float):
    # 檢查是否需要退出Blended模式
    if self.current_mode == DECConfig.MODE_BLENDED:
      if urgency_val < DECConfig.EXIT_TRIGGER_THRESHOLD:
        self.low_urgency_counter += 1
      else:
        self.low_urgency_counter = 0
      
      # 快速退出Blended
      if self.low_urgency_counter >= DECConfig.EXIT_DEBOUNCE_FRAMES:
        self.mode_confidence[DECConfig.MODE_BLENDED] *= 0.7
    else:
      self.low_urgency_counter = 0
    
    # 自然衰減
    self.mode_confidence[DECConfig.MODE_BLENDED] *= 0.95
    self.mode_confidence[DECConfig.MODE_ACC] = 1.0 - self.mode_confidence[DECConfig.MODE_BLENDED]

  def get_mode(self) -> ModeType:
    return self.current_mode


class DynamicExperimentalController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._active: bool = False
    self._frame: int = 0
    
    # 替換為 AEM 的核心組件
    self._mode_manager = ModeTransitionManager()
    self._slow_down_filter = SmoothKalmanFilter()
    self._urgency = 0.0
    self._high_urgency_counter = 0
    self._highway_suppression_active = False

    # 穩定前車檢測狀態
    self._lead_stable_counter = 0
    self._lead_unstable_counter = 0
    self._lead_confidence = 0.0
    self._last_lead_dRel = float('inf')
    self._last_lead_vRel = 0.0
    self._lead_same_target_frames = 0
    self._lead_detected_frames = 0
    self._lead_distance_history = []
    
    # 實驗模式追蹤
    self._experiment_mode_active = False
    self._experiment_enter_counter = 0
    self._last_lead_dist_when_entered = float('inf')
    
    self._mpc_fcw_crash_cnt = 0

  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._enabled = self._params.get_bool("DynamicExperimentalControl")

  def mode(self) -> str:
    return self._mode_manager.get_mode()

  def enabled(self) -> bool:
    return self._enabled

  def active(self) -> bool:
    return self._active

  def set_mpc_fcw_crash_cnt(self) -> None:
    self._mpc_fcw_crash_cnt = self._mpc.crash_cnt

  def _update_lead_stability(self, radar_msg) -> None:
    """AEM: 更新前車穩定性檢測"""
    has_lead = False
    current_lead_dRel = float('inf')
    current_lead_vRel = 0.0
    
    if radar_msg and radar_msg.leadOne and radar_msg.leadOne.status:
      has_lead = True
      current_lead_dRel = radar_msg.leadOne.dRel
      current_lead_vRel = radar_msg.leadOne.vRel
      
      if current_lead_dRel < DECConfig.LEAD_MIN_DIST:
        has_lead = False
    
    if has_lead:
      self._lead_detected_frames += 1
      distance_variance = abs(current_lead_dRel - self._last_lead_dRel)
      speed_variance = abs(current_lead_vRel - self._last_lead_vRel)
      
      is_same_target = (distance_variance < DECConfig.LEAD_STABLE_DIST_VAR and speed_variance < 2.0)
      
      if is_same_target:
        self._lead_same_target_frames += 1
        self._lead_stable_counter = min(DECConfig.LEAD_STABLE_FRAMES, self._lead_stable_counter + 1)
        self._lead_unstable_counter = max(0, self._lead_unstable_counter - 1)
      else:
        self._lead_same_target_frames = 1
        self._lead_stable_counter = 0
        self._lead_unstable_counter += 1
      
      if self._lead_same_target_frames >= DECConfig.LEAD_STABLE_FRAMES:
        self._lead_confidence = min(1.0, self._lead_stable_counter / DECConfig.LEAD_STABLE_FRAMES)
      else:
        self._lead_confidence = 0.0
      
      self._lead_distance_history.append(current_lead_dRel)
      if len(self._lead_distance_history) > DECConfig.LEAD_STABLE_FRAMES:
        self._lead_distance_history.pop(0)
      
      self._last_lead_dRel = current_lead_dRel
      self._last_lead_vRel = current_lead_vRel
      
    else:
      self._lead_detected_frames = 0
      self._lead_same_target_frames = 0
      self._lead_stable_counter = max(0, self._lead_stable_counter - 1)
      self._lead_unstable_counter += 1
      self._lead_confidence = 0.0
      self._last_lead_dRel = float('inf')
      self._last_lead_vRel = 0.0
      self._lead_distance_history = []

  def _calculate_experiment_stop_distance(self, v_ego, v_kph) -> float:
    base_expected = np.interp(v_ego, DECConfig.SLOW_DOWN_BP, DECConfig.SLOW_DOWN_DIST)
    sensitivity = np.interp(v_kph, DECConfig.SENSITIVITY_BP, DECConfig.SENSITIVITY_VALS)
    return base_expected * sensitivity * 1.1

  def _can_enter_experiment_mode(self, radar_msg, v_ego, v_kph) -> bool:
    """AEM: 檢查是否能進入實驗模式"""
    has_lead = False
    lead_dRel = float('inf')
    
    if radar_msg and radar_msg.leadOne and radar_msg.leadOne.status:
      has_lead = True
      lead_dRel = radar_msg.leadOne.dRel
      if lead_dRel < DECConfig.LEAD_MIN_DIST:
        has_lead = False
    
    if not has_lead:
      return True
    
    experiment_stop_dist = self._calculate_experiment_stop_distance(v_ego, v_kph)
    
    if self._lead_confidence < 0.8:
      return True
    
    if lead_dRel > experiment_stop_dist * 1.5:
      return True
    
    if lead_dRel < experiment_stop_dist:
      if (self._lead_same_target_frames >= DECConfig.LEAD_STABLE_FRAMES and 
          self._lead_confidence >= 0.8):
        return False
        
    return True

  def _calculate_slow_down(self, model_end_dist, v_ego, v_kph) -> None:
    """AEM: 核心急迫度計算與高速抑制"""
    base_expected = np.interp(v_ego, DECConfig.SLOW_DOWN_BP, DECConfig.SLOW_DOWN_DIST)
    sensitivity = np.interp(v_kph, DECConfig.SENSITIVITY_BP, DECConfig.SENSITIVITY_VALS)
    expected_distance = base_expected * sensitivity * 1.1

    # 綠燈/路徑通暢
    if model_end_dist > expected_distance:
      self._slow_down_filter.x = 0.0 
      self._urgency = 0.0
      self._high_urgency_counter = 0 
      return

    shortage_ratio = (expected_distance - model_end_dist) / max(1.0, expected_distance)
    raw_urgency = np.clip((shortage_ratio ** 1.5) * 2.5, 0.0, 1.2)

    # 高速抑制
    if v_kph > DECConfig.HIGHWAY_SPEED_ON:
      self._highway_suppression_active = True
    elif v_kph < DECConfig.HIGHWAY_SPEED_OFF:
      self._highway_suppression_active = False

    if self._highway_suppression_active:
      raw_urgency = min(raw_urgency, 0.4)

    self._slow_down_filter.add_data(raw_urgency)
    self._urgency = self._slow_down_filter.get_value()

  def update(self, sm: messaging.SubMaster) -> None:
    self._read_params()
    self.set_mpc_fcw_crash_cnt()

    car_state = sm['carState']
    radar_msg = sm['radarState']
    md = sm['modelV2']
    v_ego = car_state.vEgo
    v_kph = v_ego * 3.6

    # 確保資料完整才執行
    if len(md.position.z) == TRAJECTORY_SIZE:
      # [重點]：保留這美麗的錯誤，使用 z 座標取代原本 SP 的 x
      model_end_dist = md.position.z[TRAJECTORY_SIZE - 1]

      # 1. 檢測前車狀態
      self._update_lead_stability(radar_msg)

      # 2. 決定是否可進入實驗模式 (AEM + SP FCW 保留)
      if self._mpc_fcw_crash_cnt > 0:
        # SP 原生 FCW 緊急介入，強制啟動 Blended
        self._mode_manager.request_mode(DECConfig.MODE_BLENDED, confidence=1.0)
      else:
        can_enter = self._can_enter_experiment_mode(radar_msg, v_ego, v_kph)
        
        if not can_enter:
          # 無法進入實驗模式，強力降級回 ACC
          self._high_urgency_counter = 0
          self._urgency = 0.0
          self._slow_down_filter.x = 0.0
          self._mode_manager.request_mode(DECConfig.MODE_ACC, confidence=1.0)
          
          if self._experiment_mode_active:
            self._experiment_mode_active = False
            self._experiment_enter_counter = 0
            self._last_lead_dist_when_entered = float('inf')
        else:
          # 3. 綠燈恢復與紅燈減速計算
          self._calculate_slow_down(model_end_dist, v_ego, v_kph)
          
          # 4. 決策與 Debounce 機制
          CONFIRMATION_FRAMES = 5 
          
          if self._urgency > DECConfig.EXIT_TRIGGER_THRESHOLD:
            self._high_urgency_counter += 1
          else:
            self._high_urgency_counter = 0
            
          if self._high_urgency_counter >= CONFIRMATION_FRAMES:
            self._mode_manager.request_mode(DECConfig.MODE_BLENDED, confidence=min(1.0, self._urgency))
            if not self._experiment_mode_active:
              self._experiment_mode_active = True
              self._experiment_enter_counter += 1
              if radar_msg and radar_msg.leadOne and radar_msg.leadOne.status:
                self._last_lead_dist_when_entered = radar_msg.leadOne.dRel
          else:
            self._mode_manager.request_mode(DECConfig.MODE_ACC, confidence=0.9)
            if self._experiment_mode_active:
              self._experiment_mode_active = False
              self._experiment_enter_counter = 0
              self._last_lead_dist_when_entered = float('inf')

    # 更新狀態
    self._mode_manager.update(self._urgency)
    self._active = sm['selfdriveState'].experimentalMode and self._enabled
    self._frame += 1
