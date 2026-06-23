# -*- coding: utf-8 -*-
"""Step 1 안전 감지기 - 상태 머신 및 병렬 감시자 모듈.

이 모듈은 E2E LSTM 이상 감지기의 출력을 받아,
보고서 기준 Step 1(사전 궤적 이탈 감지) 단계의 안전 제어 로직을 담당한다.

구성:
    - AnomalyStateMachine : 4단계 래치 상태 머신 (Nominal→Suspect→Confirmed→EmergencyCommit)
    - RuleBasedMonitor    : 하드/소프트 규칙 기반 병렬 감시자
    - SupervisorFSM       : Confirmed → Step 2 최종 승인 로직
    - Step1SafetyDetector : 위 세 컴포넌트를 통합한 Step 1 감지기

사용 예:
    detector = Step1SafetyDetector(lstm_model, seq_length=10, input_size=6)
    result = detector.step(sensor_data)
    if result.commit_step2:
        # Step 2 (킬 스위치, 액체 덤프) 트리거
        ...
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

import numpy as np

# torch는 Step1SafetyDetector가 실제로 사용될 때만 import.
# 이 덕분에 AnomalyStateMachine / RuleBasedMonitor / SupervisorFSM은
# torch 없이도 단독 테스트 및 시뮬레이션에 사용 가능하다.


# ============================================================
# 1. 상태 정의
# ============================================================

class DetectorState(Enum):
    """Step 1 감지기의 4단계 상태."""
    NOMINAL = "nominal"                    # 정상 비행
    SUSPECT = "suspect"                    # 이상 의심 (카운터 누적)
    CONFIRMED = "confirmed"                # 이상 확정 (래치)
    EMERGENCY_COMMIT = "emergency_commit"  # Step 2 전이 신호 발송


class TriggerSource(Enum):
    """트리거 원천 - 로깅 및 디버깅용."""
    LSTM = "lstm"
    HARD_RULE = "hard_rule"       # 자유낙하, 전원 상실 등 즉시성 조건
    SOFT_RULE = "soft_rule"       # CTE 급증 등 LSTM 가속 조건
    SUPERVISOR = "supervisor"


@dataclass
class DetectionResult:
    """매 스텝의 감지 결과."""
    state: DetectorState
    lstm_probability: float
    counter: int                          # Suspect 카운터 현재값
    triggers: List[TriggerSource] = field(default_factory=list)
    rule_flags: Dict[str, bool] = field(default_factory=dict)
    commit_step2: bool = False            # True면 Step 2 트리거
    message: str = ""


@dataclass
class Step1ModelArtifact:
    """Step 1 런타임 계약을 묶어 전달하는 산출물 객체.

    모델 아키텍처 복원 정보와 별개로, 실제 추론 시점에 반드시 필요한
    seq_length, feature_columns, feature_scaler를 함께 운반한다.
    """

    model: Any
    seq_length: int
    feature_columns: List[str]
    feature_scaler: Optional[Any] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    strict_feature_columns: bool = True

    def __post_init__(self) -> None:
        self.seq_length = int(self.seq_length)
        self.feature_columns = [str(name) for name in self.feature_columns]

        if self.seq_length <= 0:
            raise ValueError("seq_length must be a positive integer.")
        if not self.feature_columns:
            raise ValueError("feature_columns must contain at least one feature.")

        if self.feature_scaler is None:
            self.feature_scaler = getattr(self.model, "feature_scaler", None)
        elif getattr(self.model, "feature_scaler", None) is None:
            # Keep model and artifact aligned so existing code paths still work.
            self.model.feature_scaler = self.feature_scaler

    @property
    def input_size(self) -> int:
        return len(self.feature_columns)

    @classmethod
    def from_model(
        cls,
        model,
        seq_length: int,
        feature_columns: Optional[List[str]] = None,
        feature_scaler=None,
        source: str = "runtime",
        metadata: Optional[Dict[str, Any]] = None,
        source_path: Optional[str] = None,
        strict_feature_columns: bool = True,
    ) -> "Step1ModelArtifact":
        return cls(
            model=model,
            seq_length=seq_length,
            feature_columns=list(feature_columns or FEATURE_ORDER),
            feature_scaler=feature_scaler,
            source=source,
            metadata=dict(metadata or {}),
            source_path=source_path,
            strict_feature_columns=strict_feature_columns,
        )


# ============================================================
# 2. 규칙 기반 병렬 감시자
# ============================================================

class RuleBasedMonitor:
    """결정론적 안전망. LSTM과 병렬로 동작하며 두 부류의 트리거를 발생.

    - Hard trigger : 즉시 Emergency Commit으로 직행 (자유낙하, 전원 상실)
    - Soft trigger : LSTM 래치 카운터를 가속 (CTE 급증, 자이로 이상)

    임계값은 보고서의 추락 감지 조건과 ALFA 데이터셋 통계를 참고하여
    기본값을 설정했으나, 실운용 전 반드시 튜닝이 필요하다.
    """

    def __init__(
        self,
        # Hard trigger 임계값
        freefall_accel_z: float = 2.0,        # m/s^2 (정상은 ≈9.8)
        freefall_hold_steps: int = 2,         # 2스텝 연속(0.2s@10Hz) 필요
        power_loss_flag_key: str = "power_lost",
        # Soft trigger 임계값
        cte_spike_threshold: float = 5.0,     # m
        gyro_attitude_loss: float = 15.0,     # rad/s (|gyro_x| or |gyro_y|)
        # 소프트 트리거가 카운터를 얼마나 가속할지
        soft_counter_boost: int = 2,
    ):
        self.freefall_accel_z = freefall_accel_z
        self.freefall_hold_steps = freefall_hold_steps
        self.power_loss_flag_key = power_loss_flag_key
        self.cte_spike_threshold = cte_spike_threshold
        self.gyro_attitude_loss = gyro_attitude_loss
        self.soft_counter_boost = soft_counter_boost

        self._freefall_counter = 0

    def reset(self) -> None:
        self._freefall_counter = 0

    def check(self, sensor: Dict[str, float]) -> Dict[str, Any]:
        """센서 데이터에서 규칙 위반을 검사한다.

        Args:
            sensor: accel_x/y/z, gyro_x/y, cross_track_error 등을 담은 dict.
                    `cross_track_error` may be a signed lateral error for the
                    LSTM. If available, `cross_track_error_filtered` or
                    `cross_track_error_abs` will be preferred for rule-based
                    CTE spike detection.
                    power_loss_flag_key로 지정된 키가 True면 전원 상실로 간주.

        Returns:
            {
              "hard_trigger": bool,
              "soft_trigger": bool,
              "flags": {"free_fall": bool, "power_loss": bool, ...},
              "soft_boost": int,
            }
        """
        flags: Dict[str, bool] = {}

        # --- Hard triggers ---
        accel_z = float(sensor.get("accel_z", 9.8))
        if accel_z < self.freefall_accel_z:
            self._freefall_counter += 1
        else:
            self._freefall_counter = 0
        flags["free_fall"] = self._freefall_counter >= self.freefall_hold_steps

        flags["power_loss"] = bool(sensor.get(self.power_loss_flag_key, False))

        hard_trigger = flags["free_fall"] or flags["power_loss"]

        # --- Soft triggers ---
        if "cross_track_error_filtered" in sensor:
            cte_for_rules = abs(float(sensor["cross_track_error_filtered"]))
        elif "cross_track_error_abs" in sensor:
            cte_for_rules = abs(float(sensor["cross_track_error_abs"]))
        else:
            cte_for_rules = abs(float(sensor.get("cross_track_error", 0.0)))
        flags["cte_spike"] = cte_for_rules > self.cte_spike_threshold

        gyro_x = abs(float(sensor.get("gyro_x", 0.0)))
        gyro_y = abs(float(sensor.get("gyro_y", 0.0)))
        flags["attitude_loss"] = (
            gyro_x > self.gyro_attitude_loss or gyro_y > self.gyro_attitude_loss
        )

        soft_trigger = flags["cte_spike"] or flags["attitude_loss"]
        soft_boost = self.soft_counter_boost if soft_trigger else 0

        return {
            "hard_trigger": hard_trigger,
            "soft_trigger": soft_trigger,
            "flags": flags,
            "soft_boost": soft_boost,
        }


# ============================================================
# 3. 상태 머신 (래치 + 히스테리시스)
# ============================================================

class AnomalyStateMachine:
    """4단계 래치 상태 머신.

    Nominal → Suspect      : LSTM 확률이 임계값 초과
    Suspect → Nominal      : 카운터 끊김 (히스테리시스 해제 조건)
    Suspect → Confirmed    : 카운터가 confirm_steps 도달 (0.5초 @10Hz)
    Confirmed → EmergencyCommit : supervisor 승인
    EmergencyCommit → *    : 수동 리셋 전까지 유지 (래치 잠금)

    Confirmed 이후에는 점수가 다시 낮아져도 자동 해제되지 않는다.
    이것이 보고서가 요구하는 "지속된 이상 확인 후 전이" 철학의 핵심.
    """

    def __init__(
        self,
        suspect_threshold: float = 0.85,       # Nominal → Suspect 점수 기준
        confirm_steps: int = 5,                # 5스텝(0.5s @10Hz) 연속 필요
        release_threshold: float = 0.60,       # 히스테리시스: 이 값 이하로 떨어지면 카운터 감소
        release_decay: int = 1,                # 점수 낮을 때 카운터에서 뺄 양
    ):
        self.suspect_threshold = suspect_threshold
        self.confirm_steps = confirm_steps
        self.release_threshold = release_threshold
        self.release_decay = release_decay

        self._state = DetectorState.NOMINAL
        self._counter = 0

    @property
    def state(self) -> DetectorState:
        return self._state

    @property
    def counter(self) -> int:
        return self._counter

    def reset(self) -> None:
        """외부에서 명시적으로 호출해야만 래치를 해제한다."""
        self._state = DetectorState.NOMINAL
        self._counter = 0

    def update(
        self,
        probability: float,
        soft_boost: int = 0,
        hard_trigger: bool = False,
    ) -> DetectorState:
        """한 스텝 진행. 상태 변경 후 새로운 상태를 반환."""

        # Hard trigger는 모든 중간 단계를 건너뛰고 즉시 Commit 직전까지 이동
        if hard_trigger and self._state != DetectorState.EMERGENCY_COMMIT:
            self._state = DetectorState.CONFIRMED
            self._counter = self.confirm_steps
            return self._state

        # Emergency Commit 래치: 외부 reset() 전까지 유지
        if self._state == DetectorState.EMERGENCY_COMMIT:
            return self._state

        # Confirmed 래치: supervisor 승인 전까지 유지 (점수가 낮아도 해제 안 함)
        if self._state == DetectorState.CONFIRMED:
            return self._state

        # Nominal / Suspect 단계의 카운터 갱신
        if probability >= self.suspect_threshold:
            self._counter += 1 + max(0, soft_boost)
        elif probability <= self.release_threshold:
            # 히스테리시스: 점수가 확실히 낮을 때만 감소
            self._counter = max(0, self._counter - self.release_decay)
        # release_threshold와 suspect_threshold 사이는 유지

        # 상태 전이 판정
        if self._counter >= self.confirm_steps:
            self._state = DetectorState.CONFIRMED
        elif self._counter > 0:
            self._state = DetectorState.SUSPECT
        else:
            self._state = DetectorState.NOMINAL

        return self._state

    def commit(self) -> None:
        """supervisor의 최종 승인 - Confirmed에서만 호출 가능."""
        if self._state == DetectorState.CONFIRMED:
            self._state = DetectorState.EMERGENCY_COMMIT


# ============================================================
# 4. Supervisor FSM (Confirmed → Step 2 승인 로직)
# ============================================================

class SupervisorFSM:
    """Confirmed 상태에서 Step 2로 넘어갈지 최종 승인하는 감독자.

    현재는 단순히 Confirmed 상태가 approve_hold_steps 동안 유지되면 승인.
    향후 확장:
      - 고도(altitude)가 낙하산 전개 최소 고도 이상이면 낙하산 포함,
        이하면 에어백만 직접 전개하는 분기
      - 배터리 전압이 이미 매우 낮으면 supercapacitor 기반 대체 전원 확인
      - 고체 화물(편심) 조건이면 multi-chamber airbag 프리셋 선택
    """

    def __init__(self, approve_hold_steps: int = 2):
        self.approve_hold_steps = approve_hold_steps
        self._confirmed_hold = 0

    def reset(self) -> None:
        self._confirmed_hold = 0

    def evaluate(
        self,
        fsm_state: DetectorState,
        hard_trigger: bool,
    ) -> bool:
        """Step 2 승인 여부 반환."""
        # Hard trigger는 즉시 승인 (자유낙하에서 대기 시간 없음)
        if hard_trigger:
            return True

        if fsm_state == DetectorState.CONFIRMED:
            self._confirmed_hold += 1
            return self._confirmed_hold >= self.approve_hold_steps

        self._confirmed_hold = 0
        return False


# ============================================================
# 5. 통합 Step 1 감지기
# ============================================================

# Keep the LSTM feature order aligned with the training data.
# `cross_track_error` is expected to be the signed lateral error signal.
FEATURE_ORDER = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y",
    "cross_track_error",
]


class Step1SafetyDetector:
    """LSTM + 규칙 감시자 + 상태 머신을 통합한 Step 1 감지기.

    실시간 루프에서 매 스텝마다 step()을 호출한다.
    commit_step2=True가 반환되면 외부 시스템이 Step 2(킬 스위치, 액체 덤프)로 전이한다.
    """

    def __init__(
        self,
        model,
        seq_length: int,
        input_size: int,
        feature_scaler=None,
        feature_columns: Optional[List[str]] = None,
        strict_feature_columns: bool = False,
        state_machine: Optional[AnomalyStateMachine] = None,
        rule_monitor: Optional[RuleBasedMonitor] = None,
        supervisor: Optional[SupervisorFSM] = None,
        device=None,
    ):
        import torch  # lazy import
        self._torch = torch

        self.model = model
        self.seq_length = seq_length
        self.input_size = input_size
        self.feature_scaler = feature_scaler or getattr(model, "feature_scaler", None)
        self.feature_columns = list(feature_columns or FEATURE_ORDER)
        self.strict_feature_columns = bool(strict_feature_columns)

        if self.input_size != len(self.feature_columns):
            raise ValueError(
                "input_size and feature_columns length must match: "
                f"{self.input_size} != {len(self.feature_columns)}"
            )

        self.state_machine = state_machine or AnomalyStateMachine()
        self.rule_monitor = rule_monitor or RuleBasedMonitor()
        self.supervisor = supervisor or SupervisorFSM()

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self._buffer = np.zeros((1, seq_length, input_size), dtype=np.float32)
        self._buffer_filled = 0

        self.model.eval()

    @classmethod
    def from_artifact(
        cls,
        artifact: Step1ModelArtifact,
        state_machine: Optional[AnomalyStateMachine] = None,
        rule_monitor: Optional[RuleBasedMonitor] = None,
        supervisor: Optional[SupervisorFSM] = None,
        device=None,
    ) -> "Step1SafetyDetector":
        return cls(
            model=artifact.model,
            seq_length=artifact.seq_length,
            input_size=artifact.input_size,
            feature_scaler=artifact.feature_scaler,
            feature_columns=artifact.feature_columns,
            strict_feature_columns=artifact.strict_feature_columns,
            state_machine=state_machine,
            rule_monitor=rule_monitor,
            supervisor=supervisor,
            device=device,
        )

    def reset(self) -> None:
        """외부에서 명시적으로 전체 리셋."""
        self.state_machine.reset()
        self.rule_monitor.reset()
        self.supervisor.reset()
        self._buffer[:] = 0.0
        self._buffer_filled = 0

    def _features_from_sensor(self, sensor: Dict[str, float]) -> np.ndarray:
        if self.strict_feature_columns:
            missing = [name for name in self.feature_columns if name not in sensor]
            if missing:
                raise KeyError(
                    "Sensor payload is missing required Step 1 features: "
                    + ", ".join(missing)
                )
        return np.array(
            [float(sensor.get(name, 0.0)) for name in self.feature_columns],
            dtype=np.float32,
        )

    def _predict(self) -> float:
        """현재 버퍼로 LSTM 확률을 얻는다.

        버퍼가 seq_length만큼 차지 않았을 때는 초기 구간의 영향이 크므로
        보수적으로 0.0을 반환해 false positive를 피한다.
        """
        if self._buffer_filled < self.seq_length:
            return 0.0

        arr = self._buffer.copy()
        if self.feature_scaler is not None:
            reshaped = arr.reshape(-1, self.input_size)
            arr = self.feature_scaler.transform(reshaped).reshape(
                1, self.seq_length, self.input_size
            )

        tensor = self._torch.tensor(arr, dtype=self._torch.float32, device=self.device)
        with self._torch.no_grad():
            prob = self.model.predict_proba(tensor).item()
        return float(prob)

    def step(self, sensor: Dict[str, float]) -> DetectionResult:
        """한 스텝 진행. sensor는 FEATURE_ORDER에 해당하는 키들을 담은 dict."""

        # 1) 슬라이딩 윈도우 버퍼 갱신
        new_row = self._features_from_sensor(sensor)
        self._buffer[0, :-1, :] = self._buffer[0, 1:, :]
        self._buffer[0, -1, :] = new_row
        self._buffer_filled = min(self._buffer_filled + 1, self.seq_length)

        # 2) LSTM 확률 계산
        lstm_prob = self._predict()

        # 3) 규칙 기반 병렬 감시자
        rule_result = self.rule_monitor.check(sensor)
        hard_trigger = rule_result["hard_trigger"]
        soft_boost = rule_result["soft_boost"]

        # 4) 상태 머신 갱신
        new_state = self.state_machine.update(
            probability=lstm_prob,
            soft_boost=soft_boost,
            hard_trigger=hard_trigger,
        )

        # 5) Supervisor 승인 검토
        approved = self.supervisor.evaluate(new_state, hard_trigger=hard_trigger)
        if approved and new_state == DetectorState.CONFIRMED:
            self.state_machine.commit()
            new_state = self.state_machine.state  # EMERGENCY_COMMIT로 갱신됨

        # 6) 트리거 원천 기록
        triggers: List[TriggerSource] = []
        if lstm_prob >= self.state_machine.suspect_threshold:
            triggers.append(TriggerSource.LSTM)
        if hard_trigger:
            triggers.append(TriggerSource.HARD_RULE)
        if rule_result["soft_trigger"]:
            triggers.append(TriggerSource.SOFT_RULE)
        if approved:
            triggers.append(TriggerSource.SUPERVISOR)

        commit_step2 = new_state == DetectorState.EMERGENCY_COMMIT

        # 7) 결과 메시지 작성
        if commit_step2:
            if hard_trigger:
                reason = [k for k, v in rule_result["flags"].items()
                          if v and k in ("free_fall", "power_loss")]
                message = f"Step 2 커밋 — 하드 트리거: {', '.join(reason)}"
            else:
                message = "Step 2 커밋 — LSTM 래치 확정 후 supervisor 승인"
        elif new_state == DetectorState.CONFIRMED:
            message = "이상 확정 — supervisor 승인 대기"
        elif new_state == DetectorState.SUSPECT:
            message = f"이상 의심 — 카운터 {self.state_machine.counter}/{self.state_machine.confirm_steps}"
        else:
            message = "정상 비행"

        return DetectionResult(
            state=new_state,
            lstm_probability=lstm_prob,
            counter=self.state_machine.counter,
            triggers=triggers,
            rule_flags=rule_result["flags"],
            commit_step2=commit_step2,
            message=message,
        )


# ============================================================
# 6. 데모 / 간이 시뮬레이터
# ============================================================

def run_step1_demo(model, seq_length: int, input_size: int, total_steps: int = 40):
    """합성 센서 스트림으로 Step 1 감지기 동작을 확인하는 데모.

    시나리오:
      step  0~14 : 정상 비행
      step 15~29 : 점진적 모터 이상 (LSTM 감지 경로)
      step 30~   : 자유낙하 발생 (하드 트리거 경로)
    """
    detector = Step1SafetyDetector(model, seq_length, input_size)

    print("--- Step 1 안전 감지기 데모 시작 ---")
    for step in range(total_steps):
        if step < 15:
            sensor = {
                "accel_x": np.random.normal(0, 0.3),
                "accel_y": np.random.normal(0, 0.3),
                "accel_z": np.random.normal(9.8, 0.3),
                "gyro_x": np.random.normal(0, 0.1),
                "gyro_y": np.random.normal(0, 0.1),
                "cross_track_error": np.random.uniform(0.1, 0.4),
            }
        elif step < 30:
            # 모터 이상: 진동 증가, CTE 점진적 상승
            severity = (step - 15) / 15.0
            sensor = {
                "accel_x": np.random.normal(3 * severity, 1.5),
                "accel_y": np.random.normal(-3 * severity, 1.5),
                "accel_z": np.random.normal(9.8 - 3 * severity, 1.0),
                "gyro_x": np.random.normal(5 * severity, 1.5),
                "gyro_y": np.random.normal(-5 * severity, 1.5),
                "cross_track_error": 0.3 + 3.0 * severity,
            }
        else:
            # 자유낙하
            sensor = {
                "accel_x": np.random.normal(0, 1.0),
                "accel_y": np.random.normal(0, 1.0),
                "accel_z": np.random.normal(0.5, 0.5),  # 자유낙하 조건
                "gyro_x": np.random.normal(0, 2.0),
                "gyro_y": np.random.normal(0, 2.0),
                "cross_track_error": 2.0,
            }

        result = detector.step(sensor)

        flags = [k for k, v in result.rule_flags.items() if v]
        flag_str = f" flags={flags}" if flags else ""
        trig_str = ",".join(t.value for t in result.triggers) or "none"

        print(
            f"[t={step*0.1:4.1f}s] state={result.state.value:17s} "
            f"p={result.lstm_probability:.2f} cnt={result.counter} "
            f"trig={trig_str}{flag_str}"
        )
        if result.commit_step2:
            print(f"   >> {result.message}")
            print("   >> Step 2 (킬 스위치 + 액체 덤프)로 제어 이관")
            break

    print("--- 데모 종료 ---")
    return detector
