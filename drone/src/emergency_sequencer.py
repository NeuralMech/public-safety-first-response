# -*- coding: utf-8 -*-
"""Step 2~5 비상 시퀀서 (EmergencySequencer).

보고서 기준 5단계 안전 시퀀스 중 Step 2~5를 담당한다.
Step 1의 commit_step2=True 신호를 받으면 즉시 실행된다.

설계 원칙:
    - 행동(actuation)은 선형 순서 : CUTOFF → DUMP → CHUTE → RADAR → AIRBAG
    - 인지/상태(perception/context)는 병렬 : 고도·하강률·레이더·하중 상태는
      매 스텝 업데이트되며 각 단계의 분기 조건에 실시간 반영
    - 시간 제약 우선 : 확인 실패가 전체 시퀀스를 멈추면 더 위험하다.
      인터락은 bounded timeout을 가진다.
    - 불확실하면 무겁게 가정 : 질량/고도 불명 시 worst-case로 행동한다.

물리 모델 (DescentPhysics):
    낙하산 전개 전  : 자유낙하 — dv/dt = +g (하강 가속)
    낙하산 전개 후  : 1차 지수 감속 — dv/dt = (v_terminal - v) / tau
                     목표 종말 속도(v_terminal ≈ 4.5 m/s)를 향해 수렴.
                     tau ≈ 2.0s → 전개 후 약 4~5초 내 90% 수렴.
    고도/TTI        : 매 스텝 실제 vertical_speed로 재계산.
    종료 조건       : altitude <= 0  또는  최대 시뮬레이션 시간 초과
                     (tick 고정값 대신 상태 기반 종료)

구성:
    - CargoState         : 현재 하중 상태 enum (TRANSIENT 포함)
    - PhysicalContext    : 병렬 센서/추정값 컨테이너
    - DescentPhysics     : 낙하산 전/후 하강 동역학 시뮬레이터
    - AirbagVentPreset   : 하중별 에어백 벤팅 파라미터
    - EmergencySequencer : Step 2~5 상태 머신
    - SequencerResult    : 매 스텝 출력

사용 예:
    seq = EmergencySequencer()
    seq.start(context)                  # EmergencyCommit 수신 시 호출
    while not seq.is_terminal:
        context = read_sensors()
        result = seq.step(context)
        actuate(result.commands)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import math


# ============================================================
# 0. 하강 동역학 모델
# ============================================================

class DescentPhysics:
    """낙하산 전/후 하강 동역학 시뮬레이터.

    낙하산 전개 전:
        자유낙하 — 중력 가속도 g=9.8 m/s² 로 하강 속도 증가.

    낙하산 전개 후:
        1차 지수 감속 모델:
            v(t) = v_terminal + (v0 - v_terminal) * exp(-Δt / tau)
        - v_terminal : 낙하산의 목표 종말 속도 (기본값 4.5 m/s)
        - tau         : 시정수. 클수록 감속이 느림.
                       130 kg 중량 기체 기준 tau ≈ 3.0s,
                       60 kg 경량 기준 tau ≈ 1.5s.
                       실제 값은 낙하산 면적·항력계수로 결정.
        - 2*tau 후 ≈90% 수렴, 5*tau 후 ≈99% 수렴.

    고도 / TTI:
        매 스텝 실제 vertical_speed 로 재계산.
        TTI = altitude / max(vertical_speed, eps)

    종료 조건:
        altitude <= 0  또는  t_total >= max_sim_seconds
    """

    GRAVITY: float = 9.8   # m/s²
    EPS:     float = 1e-6

    def __init__(
        self,
        initial_altitude: float,
        initial_vertical_speed: float,
        v_terminal: float = 4.5,
        tau_heavy: float = 3.0,
        tau_light: float = 1.5,
        max_sim_seconds: float = 120.0,
        dt: float = 0.1,
    ):
        """
        Args:
            initial_altitude        : 시뮬레이션 시작 고도 [m]
            initial_vertical_speed  : 시작 하강 속도 [m/s, 양수=하강]
            v_terminal              : 낙하산 목표 종말 속도 [m/s]
            tau_heavy               : 중량(≥100kg) 기체 감속 시정수 [s]
            tau_light               : 경량(<100kg) 기체 감속 시정수 [s]
            max_sim_seconds         : 안전 제한 — 이 시간 초과 시 종료
            dt                      : 적분 시간 간격 [s]
        """
        self.altitude        = float(initial_altitude)
        self.vertical_speed  = float(initial_vertical_speed)
        self.v_terminal      = float(v_terminal)
        self.tau_heavy       = float(tau_heavy)
        self.tau_light       = float(tau_light)
        self.max_sim_seconds = float(max_sim_seconds)
        self.dt              = float(dt)

        self._t_total        = 0.0
        self._chute_deployed = False
        self._t_since_chute  = 0.0    # 낙하산 전개 후 경과 시간

    # ── 공개 인터페이스 ────────────────────────────────────────

    @property
    def time_to_impact(self) -> float:
        """현재 속도 기준 TTI [s]."""
        return self.altitude / max(self.vertical_speed, self.EPS)

    @property
    def is_landed(self) -> bool:
        return self.altitude <= 0.0

    @property
    def is_timeout(self) -> bool:
        return self._t_total >= self.max_sim_seconds

    @property
    def is_terminal(self) -> bool:
        return self.is_landed or self.is_timeout

    def deploy_chute(self) -> None:
        """낙하산 전개 — 이 호출 이후 감속 모델로 전환."""
        if not self._chute_deployed:
            self._chute_deployed = True
            self._t_since_chute = 0.0

    def step(self, mass_kg: float = 130.0) -> None:
        """dt 만큼 물리 상태를 전진시킨다.

        Args:
            mass_kg : 현재 기체 질량. tau 선택에 사용.
        """
        if self.is_terminal:
            return

        if self._chute_deployed:
            # 낙하산 후: 1차 지수 감속
            tau = self.tau_heavy if mass_kg >= 100.0 else self.tau_light
            self._t_since_chute += self.dt
            self.vertical_speed = (
                self.v_terminal
                + (self.vertical_speed - self.v_terminal)
                * math.exp(-self.dt / tau)
            )
        else:
            # 낙하산 전: 자유낙하 가속
            self.vertical_speed += self.GRAVITY * self.dt

        # 고도 갱신
        self.altitude = max(0.0, self.altitude - self.vertical_speed * self.dt)
        self._t_total += self.dt

    def snapshot(self) -> "PhysicalContext":
        """현재 물리 상태를 PhysicalContext 로 변환."""
        return PhysicalContext(
            altitude_agl=self.altitude,
            vertical_speed=self.vertical_speed,
            time_to_impact=self.time_to_impact,
        )


# ============================================================
# 1. 하중 상태 enum
# ============================================================

class CargoState(Enum):
    """현재 기체 하중 상태.

    안전 기본 원칙: 불확실하면 UNKNOWN_HEAVY로 가정.
    에어백 벤팅에서 질량 과소추정이 질량 과대추정보다 더 위험하다.
    """
    EMPTY_LIGHT            = "empty_light"            # 화물 없음 ≈ 60 kg
    LIQUID_FULL_HEAVY      = "liquid_full_heavy"      # 액체 만재 ≈ 130 kg
    LIQUID_DUMPING_TRANSIENT = "liquid_dumping_transient"  # 덤프 진행 중
    LIQUID_DUMPED_LIGHT    = "liquid_dumped_light"    # 덤프 완료 ≈ 60 kg
    SOLID_HEAVY            = "solid_heavy"            # 고체 화물 ≈ 130 kg
    UNKNOWN_HEAVY          = "unknown_heavy"          # 불명 → 무거운 쪽 가정

    @property
    def estimated_mass_kg(self) -> float:
        """보수적(heavy-biased) 질량 추정값."""
        _MASS = {
            CargoState.EMPTY_LIGHT:             60.0,
            CargoState.LIQUID_FULL_HEAVY:      130.0,
            CargoState.LIQUID_DUMPING_TRANSIENT: 130.0,  # 전환 중 → 무거운 쪽
            CargoState.LIQUID_DUMPED_LIGHT:     60.0,
            CargoState.SOLID_HEAVY:            130.0,
            CargoState.UNKNOWN_HEAVY:          130.0,
        }
        return _MASS[self]

    @property
    def is_liquid(self) -> bool:
        return self in (
            CargoState.LIQUID_FULL_HEAVY,
            CargoState.LIQUID_DUMPING_TRANSIENT,
            CargoState.LIQUID_DUMPED_LIGHT,
        )

    @property
    def dump_allowed(self) -> bool:
        """액체 덤프가 허용되는 상태인가."""
        return self == CargoState.LIQUID_FULL_HEAVY


# ============================================================
# 2. 병렬 물리 컨텍스트 (매 스텝 외부에서 주입)
# ============================================================

@dataclass
class ImpactRiskAssessment:
    """예상 충돌 구역의 지상 위험도 평가 결과.

    planner 또는 별도 risk layer가 계산한 결과를 Step 2~5 시퀀서가 소비한다.
    핵심 철학:
        - airbag은 기본 장치가 아니라 low-risk impact zone에서만 허용되는 제한 장치
        - risk가 unknown이면 원칙적으로 fire를 허용하지 않는다
    """
    impact_risk_score: float
    impact_zone_allowed: bool
    impact_zone_label: str = "unknown"   # low_risk | high_risk | unknown
    risk_reason: str = ""

    @classmethod
    def low_risk(
        cls,
        score: float = 0.15,
        reason: str = "planner risk gate 허용",
    ) -> "ImpactRiskAssessment":
        return cls(
            impact_risk_score=float(score),
            impact_zone_allowed=True,
            impact_zone_label="low_risk",
            risk_reason=reason,
        )

    @classmethod
    def high_risk(
        cls,
        score: float = 0.85,
        reason: str = "planner risk gate 차단",
    ) -> "ImpactRiskAssessment":
        return cls(
            impact_risk_score=float(score),
            impact_zone_allowed=False,
            impact_zone_label="high_risk",
            risk_reason=reason,
        )

    @classmethod
    def unknown(
        cls,
        score: float = 1.0,
        reason: str = "impact risk assessment 미제공",
    ) -> "ImpactRiskAssessment":
        return cls(
            impact_risk_score=float(score),
            impact_zone_allowed=False,
            impact_zone_label="unknown",
            risk_reason=reason,
        )


@dataclass
class PhysicalContext:
    """시퀀서가 각 단계 분기 판단에 사용하는 물리 상태.

    Step 1 피처와 별도로, Step 2 시퀀서가 직접 받는 항법/센서 융합값.
    측정 불가 항목은 None으로 두면 worst-case 기본값이 적용된다.

    altitude_agl      : 지상 기준 고도 [m]. None이면 낙하산 전개 불가로 간주.
    vertical_speed    : 하강 속도 [m/s, 양수=하강]. None이면 최대 하강으로 간주.
    time_to_impact    : 충돌까지 남은 시간 추정 [s]. None이면 고도/하강률로 계산.
    radar_distance    : 가장 가까운 장애물까지 거리 [m].
    rotor_safe        : 로터 잠금 완료 확인. None이면 timeout 경과 후 진행.
    dump_valve_open_s : 덤프 밸브가 열려있던 누적 시간 [s].
    power_ok          : 슈퍼커패시터 전원 정상 여부.
    """
    altitude_agl:      Optional[float] = None
    vertical_speed:    Optional[float] = None  # m/s, 양수=하강
    time_to_impact:    Optional[float] = None
    radar_distance:    Optional[float] = None
    rotor_safe:        Optional[bool]  = None
    dump_valve_open_s: float           = 0.0
    power_ok:          bool            = True
    cargo_state:       CargoState      = CargoState.UNKNOWN_HEAVY
    impact_risk:       Optional[ImpactRiskAssessment] = None

    def estimated_time_to_impact(self) -> float:
        """time_to_impact가 없으면 고도/하강률로 추정. 불명 시 0.0 반환."""
        if self.time_to_impact is not None:
            return self.time_to_impact
        alt = self.altitude_agl
        vs  = self.vertical_speed
        if alt is not None and vs is not None and vs > 0:
            return alt / vs
        if alt is not None:
            return alt / 10.0   # 하강률 불명 → 10 m/s worst-case 가정
        return 0.0              # 고도 불명 → 0초, 즉 저고도 경로

    def resolved_impact_risk(self) -> ImpactRiskAssessment:
        """impact_risk가 없으면 보수적으로 unknown/high-risk로 본다."""
        if self.impact_risk is not None:
            return self.impact_risk
        return ImpactRiskAssessment.unknown()


# ============================================================
# 3. 시퀀서 단계 상태
# ============================================================

class SequencerPhase(Enum):
    IDLE              = auto()   # 대기 (Step 1 신호 전)
    ROTOR_CUTOFF      = auto()   # Step 2a: 킬 스위치 / 모터 컷오프
    RELIEF_AND_CHUTE  = auto()   # Step 2b+3: 덤프·낙하산 비블로킹 중첩 실행
    RADAR_TARGETING   = auto()   # Step 4: 전방위 레이더 감시
    AIRBAG_ARM        = auto()   # Step 5a: 에어백 무장 (물리적 팽창 없음)
    AIRBAG_PREFILL    = auto()   # Step 5b: 저압 프리팽창
    AIRBAG_FIRE       = auto()   # Step 5c: 최종 완전 팽창
    CUSHION_ONLY_DESCENT = auto()  # 위험 구역 → 보조 충격 완화체 단독 모드
    LANDED            = auto()   # 착지 완료
    ABORTED           = auto()   # 비상 중단 (전원 상실 등)


# ============================================================
# 4. 에어백 벤팅 파라미터
# ============================================================

@dataclass
class AirbagVentPreset:
    """하중별 에어백 벤팅 설정.

    실제 전자제어 밸브(ECV)에 전달될 파라미터.
    질량이 클수록 벤팅을 늦게/적게 열어 충격 흡수 시간을 확보한다.
    """
    mass_kg:             float
    vent_open_ratio:     float  # 0.0~1.0, 밸브 개방 비율
    vent_trigger_speed:  float  # m/s, 이 충돌 속도 이하에서 벤팅 시작
    chamber_pressures:   Dict[str, float] = field(default_factory=dict)
    notes:               str = ""

    @classmethod
    def from_cargo_state(cls, state: CargoState) -> "AirbagVentPreset":
        mass = state.estimated_mass_kg
        if mass <= 65.0:   # EMPTY / DUMPED
            return cls(
                mass_kg=mass,
                vent_open_ratio=0.55,
                vent_trigger_speed=4.5,
                chamber_pressures={"front": 1.0, "rear": 1.0,
                                   "left": 1.0, "right": 1.0},
                notes="경량 — 빠른 벤팅, 균등 챔버 압력",
            )
        else:              # HEAVY / TRANSIENT / UNKNOWN
            return cls(
                mass_kg=mass,
                vent_open_ratio=0.35,
                vent_trigger_speed=5.5,
                chamber_pressures={"front": 1.0, "rear": 1.0,
                                   "left": 1.0, "right": 1.0},
                notes="중량 — 느린 벤팅, 충격 흡수 시간 확보",
            )

    def adjust_for_eccentric_load(self, tilt_direction: str, boost: float = 0.2) -> None:
        """고체 화물 편심(eccentric load) 보정.

        보고서 Edge Case 3: 기울어진 방향의 챔버 압력을 높여
        multi-chamber 에어백의 자세 보정 효과를 낸다.
        """
        if tilt_direction in self.chamber_pressures:
            self.chamber_pressures[tilt_direction] = min(
                1.0, self.chamber_pressures[tilt_direction] + boost
            )


# ============================================================
# 5. 시퀀서 출력
# ============================================================

@dataclass
class SequencerResult:
    """매 스텝 시퀀서 출력."""
    phase:          SequencerPhase
    commands:       Dict[str, Any] = field(default_factory=dict)
    airbag_preset:  Optional[AirbagVentPreset] = None
    log:            str = ""
    is_terminal:    bool = False  # LANDED 또는 ABORTED


# ============================================================
# 6. 타이밍 상수 (ms 단위)
# ============================================================

class Timing:
    ROTOR_SAFE_TIMEOUT_MS:   int = 300   # 로터 안전 확인 최대 대기
    DUMP_CHUTE_STAGGER_MS:   int = 100   # 덤프 시작과 낙하산 전개 사이 micro-stagger
    DUMP_TRANSIENT_END_MS:   int = 2500  # 이 시간 이후 DUMPED_LIGHT로 전환
    CHUTE_DEPLOY_TIME_S:     float = 2.0 # 낙하산 완전 전개 소요 시간 [s]
    CHUTE_MARGIN_S:          float = 1.0 # 낙하산 전개를 위한 최소 여유 [s]
    LOW_ALT_THRESHOLD_M:     float = 5.0 # 이 고도 이하면 fast-track
    FASTTRACK_TTI_S:         float = 0.8 # 이 TTI 이하면 fast-track (저고도와 동급 위기)
    MIN_CHUTE_ALT_M:         float = 30.0  # 낙하산 전개 최소 고도 [m]
    RADAR_PREARM_ON_COMMIT:  bool  = True  # EmergencyCommit 즉시 레이더 활성화
    AIRBAG_ARM_TTI_S:        float = 1.5   # 이 TTI 이하면 arm
    AIRBAG_ARM_RADAR_M:      float = 3.0   # 이 거리 이하면 arm
    AIRBAG_PREFILL_TTI_S:    float = 0.4   # 이 TTI 이하면 prefill 허용
    AIRBAG_PREFILL_RADAR_M:  float = 1.5   # 이 거리 이하면 prefill 허용
    AIRBAG_FIRE_TTI_S:       float = 0.1   # 이 TTI 이하면 fire
    TOUCHDOWN_ALT_M:         float = 0.3   # touchdown 근사 고도
    AIRBAG_PREFILL_RATIO:    float = 0.25  # prefill 저압 비율
    CHUTE_FAILURE_CHECK_DELAY_MS: int = 1200
    CHUTE_FAILURE_DESCENT_MPS:    float = 10.0


# ============================================================
# 7. EmergencySequencer
# ============================================================

class EmergencySequencer:
    """Step 2~5 비상 시퀀서.

    외부 루프에서 매 스텝 step(context)을 호출한다.
    is_terminal이 True가 되면 루프 종료.

    Phase 구조 (최종 합의 반영):
        IDLE            : 대기
        ROTOR_CUTOFF    : 모터 컷오프 — timeout wait 있음 (저고도 fast-track 시 스킵)
        RELIEF_AND_CHUTE: 덤프 밸브 개방과 낙하산 전개를 비블로킹 중첩 실행
        RADAR_TARGETING : 레이더 감시 — 충돌 임박 감지 시 에어백 무장
        AIRBAG_ARM      : 에어백 무장. 점화/밸브 준비만, 물리적 팽창 없음
        AIRBAG_PREFILL  : 저압 프리팽창. low-risk zone일 때만 짧게 허용
        AIRBAG_FIRE     : 최종 완전 팽창
        CUSHION_ONLY_DESCENT:
                          위험 구역이라 airbag을 의도적으로 억제한 하강 상태
        LANDED / ABORTED: 종료

    핵심 원칙:
        - airbag은 "항상 켜는 보호막"이 아니라 low-risk impact zone에서만 허용
        - arm / prefill / fire를 분리해 폭발적 팽창 시간을 최소화
        - risk unknown이면 fire를 허용하지 않는다

    Fast-track 경로 (EmergencyCommit 수신 시점 판정):
        altitude_agl <= LOW_ALT_THRESHOLD_M  OR  TTI <= FASTTRACK_TTI_S 이면
        → ROTOR_CUTOFF command는 계속 내리되, timeout wait 없이 분기:
          1) low-risk 이면 AIRBAG_ARM 체인으로 직행
          2) high-risk 이고 chute 가능하면 RELIEF_AND_CHUTE로 되돌림
          3) high-risk 이고 chute도 불가하면 CUSHION_ONLY_DESCENT
    """

    def __init__(self, timing: Optional[Timing] = None):
        self._timing = timing or Timing()
        self._phase = SequencerPhase.IDLE
        self._elapsed_ms: int = 0         # 현재 phase 내 경과 ms
        self._total_elapsed_ms: int = 0   # 전체 경과 ms
        self._step_ms: int = 100          # 기본 스텝 주기 (10 Hz)
        self._chute_deployed: bool = False
        self._chute_deploy_scheduled_ms: Optional[int] = None  # micro-stagger 예약
        self._dump_valve_opened: bool = False
        self._radar_active: bool = False
        self._airbag_armed: bool = False
        self._secondary_cushion_armed: bool = False
        self._secondary_cushion_deployed: bool = False
        self._fasttrack: bool = False     # 저고도/저TTI fast-track 여부
        self._last_risk_label: str = "unknown"
        self._history: List[str] = []

    # ── 공개 인터페이스 ──────────────────────────────────────

    @property
    def phase(self) -> SequencerPhase:
        return self._phase

    @property
    def is_terminal(self) -> bool:
        return self._phase in (SequencerPhase.LANDED, SequencerPhase.ABORTED)

    def start(self, context: PhysicalContext, step_ms: int = 100) -> SequencerResult:
        """EmergencyCommit 수신 시 호출. 시퀀서를 ROTOR_CUTOFF로 전이.

        Fast-track 판정: 저고도 OR 저 TTI이면 대기/중간 단계를 접고 AIRBAG로 직행.
        """
        self._phase = SequencerPhase.ROTOR_CUTOFF
        self._elapsed_ms = 0
        self._total_elapsed_ms = 0
        self._step_ms = step_ms
        self._chute_deployed = False
        self._chute_deploy_scheduled_ms = None
        self._dump_valve_opened = False
        self._radar_active = False
        self._airbag_armed = False
        self._secondary_cushion_armed = True
        self._secondary_cushion_deployed = False
        self._last_risk_label = "unknown"
        self._history.clear()

        # Fast-track 조건 판정: 고도 OR TTI 중 하나라도 저위기 상태
        alt = context.altitude_agl
        tti = context.estimated_time_to_impact()
        low_alt = (alt is not None and alt <= self._timing.LOW_ALT_THRESHOLD_M)
        low_tti = (tti > 0 and tti <= self._timing.FASTTRACK_TTI_S)
        self._fasttrack = low_alt or low_tti

        if self._fasttrack:
            reason = []
            if low_alt: reason.append(f"alt={alt:.1f}m")
            if low_tti: reason.append(f"TTI={tti:.2f}s")
            self._log(f"FAST-TRACK 활성 ({', '.join(reason)}) — 중간 단계 스킵")

        # 레이더는 EmergencyCommit 즉시 활성화 (Codex 제안 반영)
        if self._timing.RADAR_PREARM_ON_COMMIT:
            self._radar_active = True
            self._log("레이더 선행 활성화 (EmergencyCommit 즉시)")

        return self.step(context)

    def step(self, context: PhysicalContext) -> SequencerResult:
        """한 스텝 진행. context는 최신 물리 상태."""
        if self._phase == SequencerPhase.IDLE:
            return SequencerResult(phase=self._phase, log="대기 중 — start()를 호출하세요.")

        self._elapsed_ms += self._step_ms
        self._total_elapsed_ms += self._step_ms

        # 공통: 전원 상실 감지
        if not context.power_ok:
            return self._abort("전원 상실 — 슈퍼커패시터 모드로 전환 필요")

        # 현재 phase 핸들러 라우팅
        handlers = {
            SequencerPhase.ROTOR_CUTOFF:     self._handle_rotor_cutoff,
            SequencerPhase.RELIEF_AND_CHUTE: self._handle_relief_and_chute,
            SequencerPhase.RADAR_TARGETING:  self._handle_radar_targeting,
            SequencerPhase.AIRBAG_ARM:       self._handle_airbag_arm,
            SequencerPhase.AIRBAG_PREFILL:   self._handle_airbag_prefill,
            SequencerPhase.AIRBAG_FIRE:      self._handle_airbag_fire,
            SequencerPhase.CUSHION_ONLY_DESCENT: self._handle_cushion_only_descent,
            SequencerPhase.LANDED:           self._handle_landed,
        }
        handler = handlers.get(self._phase)
        if handler is None:
            return SequencerResult(phase=self._phase, is_terminal=True,
                                   log="알 수 없는 phase")
        return handler(context)

    # ── Phase 핸들러 ─────────────────────────────────────────

    def _handle_rotor_cutoff(self, context: PhysicalContext) -> SequencerResult:
        """Step 2a: motor cutoff and secondary cushion arm hold."""
        commands = self._base_commands(context)

        if self._fasttrack:
            risk = self._resolve_impact_risk(context)
            commands.update(self._risk_command_fields(risk))
            self._deploy_secondary_cushion(commands, reason="fast-track")
            self._log(
                f"Fast-track branch -> risk={risk.impact_zone_label} "
                f"score={risk.impact_risk_score:.2f}"
            )
            self._transition(SequencerPhase.AIRBAG_ARM)
            preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
            armed = self._arm_commands(context, preset, risk)
            armed.update(commands)
            return SequencerResult(
                phase=self._phase,
                commands=armed,
                airbag_preset=preset,
                log="Fast-track: deployed secondary cushion and entered AIRBAG_ARM",
            )

        rotor_confirmed = (
            context.rotor_safe is True
            or self._elapsed_ms >= self._timing.ROTOR_SAFE_TIMEOUT_MS
        )

        if rotor_confirmed:
            reason = (
                "confirmed"
                if context.rotor_safe
                else f"timeout {self._timing.ROTOR_SAFE_TIMEOUT_MS}ms"
            )
            self._log(f"Rotor safe {reason} -> RELIEF_AND_CHUTE")
            self._transition(SequencerPhase.RELIEF_AND_CHUTE)

        return SequencerResult(
            phase=self._phase,
            commands=commands,
            log=f"Rotor cutoff in progress (elapsed={self._elapsed_ms}ms)",
        )

    def _handle_relief_and_chute(self, context: PhysicalContext) -> SequencerResult:
        """Step 2b + Step 3: dump relief and parachute sequence."""
        commands: Dict[str, Any] = {**self._base_commands(context)}
        cargo = context.cargo_state

        if not self._dump_valve_opened:
            if cargo.dump_allowed:
                self._dump_valve_opened = True
                self._chute_deploy_scheduled_ms = (
                    self._total_elapsed_ms + self._timing.DUMP_CHUTE_STAGGER_MS
                )
                self._log(
                    f"Dump valve opened -> chute schedule +{self._timing.DUMP_CHUTE_STAGGER_MS}ms"
                )
            else:
                self._chute_deploy_scheduled_ms = self._total_elapsed_ms
                self._log(f"Dump not allowed ({cargo.value}) -> immediate chute scheduling")

        if self._dump_valve_opened:
            dump_elapsed = context.dump_valve_open_s * 1000
            commands["dump_valve_open"] = dump_elapsed < self._timing.DUMP_TRANSIENT_END_MS

        scheduled = self._chute_deploy_scheduled_ms
        time_reached = scheduled is not None and self._total_elapsed_ms >= scheduled

        if time_reached and not self._chute_deployed:
            alt = context.altitude_agl
            tti = context.estimated_time_to_impact()
            min_tti = self._timing.CHUTE_DEPLOY_TIME_S + self._timing.CHUTE_MARGIN_S
            low_alt = alt is not None and alt <= self._timing.LOW_ALT_THRESHOLD_M
            time_insufficient = tti < min_tti
            risk = self._resolve_impact_risk(context)

            if low_alt or time_insufficient:
                reason = "low altitude" if low_alt else f"TTI {tti:.1f}s < {min_tti:.1f}s"
                self._deploy_secondary_cushion(commands, reason=f"chute skipped ({reason})")
                self._transition(SequencerPhase.AIRBAG_ARM)
                preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
                armed = self._arm_commands(context, preset, risk)
                armed.update(commands)
                return SequencerResult(
                    phase=self._phase,
                    commands=armed,
                    airbag_preset=preset,
                    log=f"Chute skipped ({reason}) -> secondary cushion deploy + AIRBAG_ARM",
                )

            commands["parachute_deploy"] = True
            self._chute_deployed = True
            self._log(f"Parachute deploy -> alt={alt:.1f}m TTI={tti:.1f}s")
            self._transition(SequencerPhase.RADAR_TARGETING)
            return SequencerResult(
                phase=self._phase,
                commands=commands,
                log="Parachute deployed -> RADAR_TARGETING",
            )

        return SequencerResult(
            phase=self._phase,
            commands=commands,
            log=f"Relief/chute waiting (dump_open={self._dump_valve_opened}, sched={scheduled}ms)",
        )

    def _handle_radar_targeting(self, context: PhysicalContext) -> SequencerResult:
        """Step 4: radar targeting and chute failure monitoring."""
        commands = self._base_commands(context)
        tti = context.estimated_time_to_impact()
        radar_dist = context.radar_distance
        risk = self._resolve_impact_risk(context)
        commands.update(self._risk_command_fields(risk))

        if self._landing_detected(context):
            self._transition(SequencerPhase.LANDED)
            return SequencerResult(
                phase=SequencerPhase.LANDED,
                commands=commands,
                is_terminal=True,
                log="Touchdown reached during radar targeting",
            )

        chute_failed = (
            self._chute_deployed
            and context.vertical_speed is not None
            and context.vertical_speed > self._timing.CHUTE_FAILURE_DESCENT_MPS
            and self._elapsed_ms >= self._timing.CHUTE_FAILURE_CHECK_DELAY_MS
        )
        if chute_failed:
            self._deploy_secondary_cushion(commands, reason="chute failure detected")
            self._transition(SequencerPhase.AIRBAG_ARM)
            preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
            armed = self._arm_commands(context, preset, risk)
            armed.update(commands)
            return SequencerResult(
                phase=self._phase,
                commands=armed,
                airbag_preset=preset,
                log="Chute failure detected -> secondary cushion deploy + AIRBAG_ARM",
            )

        arm_condition = (
            tti <= self._timing.AIRBAG_ARM_TTI_S
            or (radar_dist is not None and radar_dist <= self._timing.AIRBAG_ARM_RADAR_M)
        )
        if arm_condition:
            preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
            self._log(
                f"AIRBAG_ARM ready -> risk={risk.impact_zone_label} "
                f"score={risk.impact_risk_score:.2f}"
            )
            self._transition(SequencerPhase.AIRBAG_ARM)
            return SequencerResult(
                phase=self._phase,
                commands=self._arm_commands(context, preset, risk),
                airbag_preset=preset,
                log=f"AIRBAG_ARM entered (TTI={tti:.1f}s, risk={risk.impact_zone_label})",
            )

        return SequencerResult(
            phase=self._phase,
            commands=commands,
            log=f"Radar targeting active (TTI={tti:.1f}s, dist={radar_dist}m, risk={risk.impact_zone_label})",
        )

    def _handle_airbag_arm(self, context: PhysicalContext) -> SequencerResult:
        """Airbag armed state with optional cushion-only fallback."""
        preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
        risk = self._resolve_impact_risk(context)
        commands = self._arm_commands(context, preset, risk)

        if self._landing_detected(context):
            self._transition(SequencerPhase.LANDED)
            return SequencerResult(
                phase=SequencerPhase.LANDED,
                commands=commands,
                airbag_preset=preset,
                is_terminal=True,
                log="Touchdown completed from AIRBAG_ARM state",
            )

        if risk.impact_zone_allowed and self._should_fire(context):
            self._transition(SequencerPhase.AIRBAG_FIRE)
            return self._fire_result(context, preset, risk)

        if risk.impact_zone_allowed and self._should_prefill(context):
            self._transition(SequencerPhase.AIRBAG_PREFILL)
            commands["airbag_prefill"] = True
            commands["prefill_ratio"] = self._timing.AIRBAG_PREFILL_RATIO
            self._log("Low-risk confirmed -> AIRBAG_PREFILL start")
            return SequencerResult(
                phase=self._phase,
                commands=commands,
                airbag_preset=preset,
                log="Airbag prefill started",
            )

        if (not risk.impact_zone_allowed) and (
            self._should_prefill(context) or self._should_fire(context)
        ):
            self._deploy_secondary_cushion(commands, reason="airbag inhibited by risk gate")
            self._transition(SequencerPhase.CUSHION_ONLY_DESCENT)
            commands["airbag_inhibited"] = True
            return SequencerResult(
                phase=self._phase,
                commands=commands,
                airbag_preset=preset,
                log="High-risk gate -> cushion-only descent mode",
            )

        return SequencerResult(
            phase=self._phase,
            commands=commands,
            airbag_preset=preset,
            log=f"Airbag armed, awaiting trigger (risk={risk.impact_zone_label})",
        )

    def _handle_airbag_prefill(self, context: PhysicalContext) -> SequencerResult:
        """Low-pressure prefill stage."""
        preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
        risk = self._resolve_impact_risk(context)
        commands = self._arm_commands(context, preset, risk)
        commands["airbag_prefill"] = True
        commands["prefill_ratio"] = self._timing.AIRBAG_PREFILL_RATIO

        if not risk.impact_zone_allowed:
            self._deploy_secondary_cushion(commands, reason="risk changed during prefill")
            self._transition(SequencerPhase.CUSHION_ONLY_DESCENT)
            commands["airbag_inhibited"] = True
            return SequencerResult(
                phase=self._phase,
                commands=commands,
                airbag_preset=preset,
                log="Risk escalated during prefill -> cushion-only descent",
            )

        if self._should_fire(context) or self._landing_detected(context):
            self._transition(SequencerPhase.AIRBAG_FIRE)
            return self._fire_result(context, preset, risk)

        return SequencerResult(
            phase=self._phase,
            commands=commands,
            airbag_preset=preset,
            log="Airbag prefill active",
        )

    def _handle_airbag_fire(self, context: PhysicalContext) -> SequencerResult:
        """최종 완전 팽창."""
        preset = AirbagVentPreset.from_cargo_state(context.cargo_state)
        risk = self._resolve_impact_risk(context)
        return self._fire_result(context, preset, risk)

    def _handle_cushion_only_descent(
        self,
        context: PhysicalContext,
    ) -> SequencerResult:
        """Airbag inhibited, secondary cushion only harm mitigation path."""
        risk = self._resolve_impact_risk(context)
        commands = self._base_commands(context)
        commands.update(self._risk_command_fields(risk))
        commands["airbag_inhibited"] = True

        if not self._secondary_cushion_deployed:
            self._deploy_secondary_cushion(commands, reason="entered cushion-only mode")

        if self._landing_detected(context):
            self._transition(SequencerPhase.LANDED)
            return SequencerResult(
                phase=SequencerPhase.LANDED,
                commands=commands,
                is_terminal=True,
                log="Cushion-only descent finished with touchdown",
            )

        return SequencerResult(
            phase=self._phase,
            commands=commands,
            log=(
                f"Cushion-only descent active (risk={risk.impact_zone_label}, "
                f"score={risk.impact_risk_score:.2f})"
            ),
        )

    def _handle_landed(self, context: PhysicalContext) -> SequencerResult:
        return SequencerResult(
            phase=SequencerPhase.LANDED,
            is_terminal=True,
            log="착지 완료 상태 유지",
        )

    # ── 내부 유틸 ─────────────────────────────────────────────

    def _base_commands(self, context: PhysicalContext) -> Dict[str, Any]:
        return {
            "motor_cutoff": True,
            "rotor_lock": True,
            "radar_active": self._radar_active,
            "parachute_deployed": self._chute_deployed,
            "secondary_cushion_arm": self._secondary_cushion_armed,
            "secondary_cushion_deployed": self._secondary_cushion_deployed,
        }

    def _risk_command_fields(
        self,
        risk: ImpactRiskAssessment,
    ) -> Dict[str, Any]:
        self._last_risk_label = risk.impact_zone_label
        return {
            "impact_risk_score": risk.impact_risk_score,
            "impact_zone_allowed": risk.impact_zone_allowed,
            "impact_zone_label": risk.impact_zone_label,
            "impact_risk_reason": risk.risk_reason,
        }

    def _resolve_impact_risk(
        self,
        context: PhysicalContext,
    ) -> ImpactRiskAssessment:
        return context.resolved_impact_risk()

    def _is_chute_viable(self, context: PhysicalContext) -> bool:
        alt = context.altitude_agl
        tti = context.estimated_time_to_impact()
        min_tti = self._timing.CHUTE_DEPLOY_TIME_S + self._timing.CHUTE_MARGIN_S
        return (
            alt is not None
            and alt >= self._timing.MIN_CHUTE_ALT_M
            and tti >= min_tti
        )

    def _should_prefill(self, context: PhysicalContext) -> bool:
        tti = context.estimated_time_to_impact()
        radar_dist = context.radar_distance
        return (
            tti <= self._timing.AIRBAG_PREFILL_TTI_S
            or (
                radar_dist is not None
                and radar_dist <= self._timing.AIRBAG_PREFILL_RADAR_M
            )
        )

    def _should_fire(self, context: PhysicalContext) -> bool:
        tti = context.estimated_time_to_impact()
        return tti <= self._timing.AIRBAG_FIRE_TTI_S or self._landing_detected(context)

    def _landing_detected(self, context: PhysicalContext) -> bool:
        landed_by_alt = (
            context.altitude_agl is not None
            and context.altitude_agl <= self._timing.TOUCHDOWN_ALT_M
        )
        landed_by_tti = context.estimated_time_to_impact() <= 0.0
        return landed_by_alt or landed_by_tti

    def _arm_commands(
        self,
        context: PhysicalContext,
        preset: AirbagVentPreset,
        risk: ImpactRiskAssessment,
    ) -> Dict[str, Any]:
        self._airbag_armed = True
        commands = self._base_commands(context)
        commands.update(self._risk_command_fields(risk))
        commands["airbag_armed"] = True
        commands["airbag_preset"] = {
            "vent_open_ratio": preset.vent_open_ratio,
            "vent_trigger_speed": preset.vent_trigger_speed,
            "chamber_pressures": preset.chamber_pressures.copy(),
        }
        return commands

    def _deploy_secondary_cushion(
        self,
        commands: Dict[str, Any],
        reason: str,
    ) -> None:
        if not self._secondary_cushion_armed:
            self._secondary_cushion_armed = True
        if not self._secondary_cushion_deployed:
            self._secondary_cushion_deployed = True
            commands["secondary_cushion_deploy"] = True
            self._log(f"보조 충격 완화체 deploy ({reason})")

    def _fire_result(
        self,
        context: PhysicalContext,
        preset: AirbagVentPreset,
        risk: ImpactRiskAssessment,
    ) -> SequencerResult:
        commands = self._arm_commands(context, preset, risk)
        commands["airbag_fire"] = True
        commands["airbag_deploy"] = True  # 하위 호환 alias
        commands["vent_open_ratio"] = preset.vent_open_ratio
        commands["vent_trigger_speed"] = preset.vent_trigger_speed
        commands["chamber_pressures"] = preset.chamber_pressures.copy()

        if self._chute_deployed and context.estimated_time_to_impact() <= 0.2:
            commands["parachute_release"] = True
            self._log("낙하산 연결 해제 — 착지 끌림 방지")

        if self._landing_detected(context):
            self._log("착지 완료")
            self._transition(SequencerPhase.LANDED)
            return SequencerResult(
                phase=SequencerPhase.LANDED,
                commands=commands,
                airbag_preset=preset,
                is_terminal=True,
                log="에어백 완전 팽창 후 착지 완료",
            )

        return SequencerResult(
            phase=SequencerPhase.AIRBAG_FIRE,
            commands=commands,
            airbag_preset=preset,
            log=f"에어백 완전 팽창 유지 — risk={risk.impact_zone_label}",
        )

    def _transition(self, new_phase: SequencerPhase) -> None:
        self._log(f"phase 전이: {self._phase.name} → {new_phase.name}")
        self._phase = new_phase
        self._elapsed_ms = 0

    def _abort(self, reason: str) -> SequencerResult:
        self._log(f"ABORT: {reason}")
        self._phase = SequencerPhase.ABORTED
        return SequencerResult(
            phase=SequencerPhase.ABORTED,
            commands={
                "motor_cutoff": True,
                "rotor_lock": True,
                "radar_active": self._radar_active,
                "parachute_deployed": self._chute_deployed,
                "secondary_cushion_arm": self._secondary_cushion_armed,
                "secondary_cushion_deployed": self._secondary_cushion_deployed,
                "airbag_inhibited": True,
            },
            is_terminal=True,
            log=f"비상 중단: {reason}",
        )

    def _log(self, msg: str) -> None:
        entry = f"[t={self._total_elapsed_ms}ms] {msg}"
        self._history.append(entry)

    @property
    def history(self) -> List[str]:
        return list(self._history)


# ============================================================
# 8. 시나리오 시뮬레이터
# ============================================================

def _make_context(
    alt: float,
    vs: float,
    cargo: CargoState = CargoState.LIQUID_FULL_HEAVY,
    rotor_safe: Optional[bool] = None,
    dump_s: float = 0.0,
    radar: Optional[float] = None,
    power: bool = True,
    impact_risk: Optional[ImpactRiskAssessment] = None,
) -> PhysicalContext:
    return PhysicalContext(
        altitude_agl=alt,
        vertical_speed=vs,
        radar_distance=radar,
        rotor_safe=rotor_safe,
        dump_valve_open_s=dump_s,
        power_ok=power,
        cargo_state=cargo,
        impact_risk=impact_risk,
    )


def run_sequencer_demo():
    """세 가지 시나리오로 EmergencySequencer + DescentPhysics 동작 확인.

    종료 조건 (tick 고정 제거):
        - physics.is_landed  (altitude <= 0)
        - seq.is_terminal    (LANDED or ABORTED)
        - physics.is_timeout (최대 시뮬레이션 시간 초과)
    """

    scenarios = [
        {
            "name": "Scenario A: altitude 80m, liquid cargo, low-risk full chain",
            "initial_alt": 80.0,
            "initial_vs":   5.0,
            "cargo": CargoState.LIQUID_FULL_HEAVY,
            "impact_risk": ImpactRiskAssessment.low_risk(
                reason="open field / low-risk impact zone"
            ),
        },
        {
            "name": "Scenario B: altitude 4m, solid cargo, high-risk fast-track",
            "initial_alt":  4.0,
            "initial_vs":   8.0,
            "cargo": CargoState.SOLID_HEAVY,
            "impact_risk": ImpactRiskAssessment.high_risk(
                reason="pedestrian corridor / airbag inhibit"
            ),
        },
        {
            "name": "Scenario C: altitude 60m, solid cargo, low-risk chute plus airbag",
            "initial_alt": 60.0,
            "initial_vs":   4.0,
            "cargo": CargoState.SOLID_HEAVY,
            "impact_risk": ImpactRiskAssessment.low_risk(
                score=0.10,
                reason="planned low-risk impact zone"
            ),
        },
        {
            "name": "Scenario D: altitude 15m, rapid descent, high-risk cushion-only",
            "initial_alt": 15.0,
            "initial_vs":  25.0,   # TTI=0.6s -> fast-track
            "cargo": CargoState.LIQUID_FULL_HEAVY,
            "impact_risk": ImpactRiskAssessment.high_risk(
                score=0.95,
                reason="dense urban core / no airbag fire"
            ),
        },
    ]

    for sc in scenarios:
        print("\n" + "=" * 65)
        print(sc["name"])
        print("=" * 65)

        step_ms  = 100
        dt       = step_ms / 1000.0
        cargo    = sc["cargo"]
        dump_s   = 0.0

        physics = DescentPhysics(
            initial_altitude=sc["initial_alt"],
            initial_vertical_speed=sc["initial_vs"],
            max_sim_seconds=120.0,
            dt=dt,
        )
        seq = EmergencySequencer()

        # 첫 스텝: EmergencyCommit 수신
        ctx = physics.snapshot()
        ctx.cargo_state = cargo
        ctx.rotor_safe = None
        ctx.dump_valve_open_s = dump_s
        ctx.radar_distance = physics.altitude if physics.altitude < 10.0 else None
        ctx.impact_risk = sc["impact_risk"]

        result = seq.start(ctx, step_ms=step_ms)
        _print_result(0, result, physics)

        tick = 1
        while not seq.is_terminal and not physics.is_terminal:
            # ── 물리 전진 ──────────────────────────────────────
            # 낙하산 전개 명령이 이전 스텝에서 내려졌으면 모델 전환
            if result.commands.get("parachute_deploy") and not physics._chute_deployed:
                physics.deploy_chute()

            physics.step(mass_kg=cargo.estimated_mass_kg)

            # ── 덤프 상태 갱신 ─────────────────────────────────
            if result.commands.get("dump_valve_open"):
                dump_s += dt
                if dump_s >= Timing.DUMP_TRANSIENT_END_MS / 1000.0:
                    cargo = CargoState.LIQUID_DUMPED_LIGHT

            # ── 컨텍스트 조립 ──────────────────────────────────
            ctx = physics.snapshot()
            ctx.cargo_state = cargo
            ctx.rotor_safe = (tick >= 3)   # 3스텝(0.3s) 후 로터 안전 확인
            ctx.dump_valve_open_s = dump_s
            # 레이더: 10m 이하에서 장애물 거리 = 고도로 근사
            ctx.radar_distance = physics.altitude if physics.altitude < 10.0 else None
            ctx.impact_risk = sc["impact_risk"]

            result = seq.step(ctx)
            _print_result(tick * step_ms, result, physics)
            tick += 1

        # ── 종료 이유 ──────────────────────────────────────────
        if physics.is_landed:
            impact_tag = (
                "안전 착지 ✓" if physics.vertical_speed <= 5.5
                else f"경착지 ⚠ ({physics.vertical_speed:.1f} m/s)"
            )
            print(f"\n  >> 착지 완료 — {impact_tag}")
        elif physics.is_timeout:
            print("\n  >> 최대 시뮬레이션 시간 초과 (물리 모델 확인 필요)")

        print("\n--- 시퀀서 로그 ---")
        for entry in seq.history:
            print(" ", entry)


def _print_result(
    t_ms: int,
    result: SequencerResult,
    physics: Optional["DescentPhysics"] = None,
) -> None:
    cmds = [k for k, v in result.commands.items() if v is True]
    phys_str = ""
    if physics is not None:
        phys_str = (
            f"  alt={physics.altitude:6.1f}m "
            f"vs={physics.vertical_speed:5.2f}m/s "
            f"TTI={physics.time_to_impact:5.1f}s"
        )
    print(
        f"  [t={t_ms:5d}ms] phase={result.phase.name:20s}"
        f"{phys_str}  {result.log}"
    )


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run_sequencer_demo()
