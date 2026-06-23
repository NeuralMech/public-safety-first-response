import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Public_Safety_EndToEnd_Demo.ipynb"


def lines(text: str):
    return [line + "\n" for line in dedent(text).strip("\n").splitlines()]


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebook():
    cells = [
        md_cell(
            """
            # Public Safety End-to-End Demo

            이 노트북은 `지켜줘, 골든-타임` 시스템의 상위 서비스 흐름을 코드로 재현한다.

            핵심 흐름은 다음과 같다.

            1. CCTV 또는 외부 신고가 위험 신호를 생성한다.
            2. 시계열 누적 검증과 위험 점수 산출을 통해 권고 단계를 만든다.
            3. 인간 관제자가 최종 출동 여부를 승인한다.
            4. 승인된 사건 메시지를 드론 임무 메시지로 변환한다.

            이 노트북은 탐지 모델 학습이나 실제 비행 제어가 아니라, 계층 간 메시지와 책임 경계를 확인하기 위한 통합 데모이다.
            """
        ),
        md_cell(
            """
            ## System Contract

            - CCTV 조기 인지 계층: 위험 객체 탐지와 시계열 누적 검증
            - 관제 판단 계층: 위험 점수, 권고 단계, 관제자 승인
            - 드론 초기 대응 계층: 목적지, 우선순위, 경로 정책, 탑재 키트 설정
            - 드론 내부 안전 계층: 지상 피해 기반 경로 계획, 이상 감지, 비상 대응 시퀀스

            사건 메시지와 임무 메시지는 JSON으로 표현한다. 설계 목표는 CCTV 조기 인지부터 드론 임무 수신까지의 통신 경로 처리 지연을 3초 이내로 유지하는 것이다.
            """
        ),
        code_cell(
            """
            from dataclasses import dataclass, asdict
            from typing import Dict, List, Optional, Tuple
            import json
            import pandas as pd
            """
        ),
        code_cell(
            """
            @dataclass
            class DetectionSignal:
                class_name: str
                confidence: float


            @dataclass
            class EventContext:
                source: str
                city_area: str
                location_label: str
                latitude: float
                longitude: float
                external_report: bool = False
                operator_marked_hotspot: bool = False


            @dataclass
            class TemporalState:
                fire_count: int = 0
                knife_count: int = 0
                fire_threshold: int = 4
                knife_threshold: int = 3

                @property
                def fire_alarm(self) -> bool:
                    return self.fire_count >= self.fire_threshold

                @property
                def knife_alarm(self) -> bool:
                    return self.knife_count >= self.knife_threshold


            @dataclass
            class EventMessage:
                event_id: str
                source: str
                city_area: str
                location_label: str
                coordinates: Tuple[float, float]
                detected_objects: List[Dict]
                temporal_state: Dict
                risk_score: float
                recommendation_level: str
                recommendation_label: str
                dispatch_recommended: bool
                human_approval_required: bool
                evidence: List[str]


            @dataclass
            class OperatorDecision:
                event_id: str
                approved: bool
                operator_id: str
                note: str


            @dataclass
            class DroneMissionMessage:
                mission_id: str
                event_id: str
                target_location: str
                target_coordinates: Tuple[float, float]
                priority: str
                recommended_mode: str
                payload_kit: str
                route_policy: str
                communication: Dict[str, str]
                target_message_latency_s: float
            """
        ),
        code_cell(
            """
            LEVEL_LABELS = {
                "observe": "관찰",
                "review": "확인",
                "dispatch": "출동 권고",
                "urgent": "긴급 출동 권고",
            }


            def max_conf(detections: List[DetectionSignal], class_name: str) -> float:
                values = [d.confidence for d in detections if d.class_name == class_name]
                return max(values) if values else 0.0


            def classify_level(score: float) -> str:
                if score >= 0.85:
                    return "urgent"
                if score >= 0.60:
                    return "dispatch"
                if score >= 0.30:
                    return "review"
                return "observe"


            def score_event(
                detections: List[DetectionSignal],
                temporal: TemporalState,
                context: EventContext,
            ) -> tuple[float, str, List[str]]:
                fire_conf = max_conf(detections, "fire")
                knife_conf = max_conf(detections, "knife")
                person_conf = max_conf(detections, "person")

                evidence = []
                fire_score = 0.0
                knife_score = 0.0

                if temporal.fire_alarm:
                    fire_score = 0.55 + 0.35 * fire_conf
                    evidence.append("화재 징후 누적")
                elif fire_conf >= 0.25:
                    fire_score = 0.15 + 0.35 * fire_conf
                    evidence.append("화재 단발 탐지")

                if temporal.knife_alarm:
                    knife_score = 0.45 + 0.40 * knife_conf
                    evidence.append("흉기 징후 누적")
                elif knife_conf >= 0.20:
                    knife_score = 0.10 + 0.35 * knife_conf
                    evidence.append("흉기 단발 탐지")

                score = max(fire_score, knife_score)

                if person_conf > 0.0 and knife_conf > 0.0:
                    score += 0.05
                    evidence.append("사람과 흉기 동시 탐지")

                if context.external_report:
                    score += 0.15
                    evidence.append("외부 신고 확인")

                if context.operator_marked_hotspot:
                    score += 0.10
                    evidence.append("관제자 지정 주의 구역")

                if detections and max(fire_conf, knife_conf) < 0.35 and not context.external_report:
                    score -= 0.10
                    evidence.append("낮은 신뢰도의 단발 탐지")

                score = round(max(0.0, min(1.0, score)), 3)
                level = classify_level(score)
                return score, level, evidence or ["유의미한 위험 신호 없음"]
            """
        ),
        code_cell(
            """
            def build_event_message(
                event_id: str,
                detections: List[DetectionSignal],
                temporal: TemporalState,
                context: EventContext,
            ) -> EventMessage:
                score, level, evidence = score_event(detections, temporal, context)
                return EventMessage(
                    event_id=event_id,
                    source=context.source,
                    city_area=context.city_area,
                    location_label=context.location_label,
                    coordinates=(context.latitude, context.longitude),
                    detected_objects=[asdict(d) for d in detections],
                    temporal_state={
                        "fire_count": temporal.fire_count,
                        "knife_count": temporal.knife_count,
                        "fire_alarm": temporal.fire_alarm,
                        "knife_alarm": temporal.knife_alarm,
                    },
                    risk_score=score,
                    recommendation_level=level,
                    recommendation_label=LEVEL_LABELS[level],
                    dispatch_recommended=level in {"dispatch", "urgent"},
                    human_approval_required=True,
                    evidence=evidence,
                )


            def select_payload_kit(event: EventMessage) -> str:
                classes = {item["class_name"] for item in event.detected_objects}
                if "fire" in classes:
                    return "fire_disaster_kit"
                if "knife" in classes:
                    return "security_standoff_observation"
                return "emergency_first_aid_kit"


            def select_mission_mode(event: EventMessage) -> str:
                if event.recommendation_level == "urgent":
                    return "rapid_arrival_and_live_relay"
                if event.recommendation_level == "dispatch":
                    return "visual_confirmation_and_information_relay"
                return "standby"


            def build_mission_message(
                event: EventMessage,
                decision: OperatorDecision,
                launch_site: str = "sejong-5-1-rooftop-station",
            ) -> Optional[DroneMissionMessage]:
                if not decision.approved:
                    return None

                priority = "P1" if event.recommendation_level == "urgent" else "P2"
                return DroneMissionMessage(
                    mission_id=f"mission-{event.event_id}",
                    event_id=event.event_id,
                    target_location=event.location_label,
                    target_coordinates=event.coordinates,
                    priority=priority,
                    recommended_mode=select_mission_mode(event),
                    payload_kit=select_payload_kit(event),
                    route_policy="hard_constraints + ground_risk_aware_astar + emergency_impact_risk_gate",
                    communication={
                        "control_network": "existing_cctv_fiber_network",
                        "drone_network": "private_5g_or_equivalent_low_latency_link",
                        "message_format": "json",
                        "launch_site": launch_site,
                    },
                    target_message_latency_s=3.0,
                )
            """
        ),
        code_cell(
            """
            scenarios = [
                {
                    "event_id": "evt-001",
                    "context": EventContext(
                        source="smart_cctv",
                        city_area="Sejong 5-1 Living Area",
                        location_label="school-zone intersection",
                        latitude=36.4961,
                        longitude=127.2664,
                        external_report=False,
                        operator_marked_hotspot=True,
                    ),
                    "temporal": TemporalState(fire_count=0, knife_count=1),
                    "detections": [DetectionSignal("knife", 0.41)],
                    "operator_approved": False,
                    "operator_note": "단발 탐지이므로 영상 재확인",
                },
                {
                    "event_id": "evt-002",
                    "context": EventContext(
                        source="smart_cctv + citizen_report",
                        city_area="Sejong 5-1 Living Area",
                        location_label="commercial block B",
                        latitude=36.4970,
                        longitude=127.2701,
                        external_report=True,
                        operator_marked_hotspot=True,
                    ),
                    "temporal": TemporalState(fire_count=0, knife_count=3),
                    "detections": [DetectionSignal("person", 0.72), DetectionSignal("knife", 0.76)],
                    "operator_approved": True,
                    "operator_note": "외부 신고와 누적 탐지가 일치하여 출동 승인",
                },
                {
                    "event_id": "evt-003",
                    "context": EventContext(
                        source="smart_cctv + fire_report",
                        city_area="Sejong 5-1 Living Area",
                        location_label="residential block C",
                        latitude=36.4992,
                        longitude=127.2722,
                        external_report=True,
                        operator_marked_hotspot=False,
                    ),
                    "temporal": TemporalState(fire_count=4, knife_count=0),
                    "detections": [DetectionSignal("fire", 0.88)],
                    "operator_approved": True,
                    "operator_note": "화재 징후 누적 및 신고 확인",
                },
            ]

            rows = []
            event_messages = []
            mission_messages = []

            for item in scenarios:
                event = build_event_message(
                    item["event_id"],
                    item["detections"],
                    item["temporal"],
                    item["context"],
                )
                decision = OperatorDecision(
                    event_id=event.event_id,
                    approved=bool(item["operator_approved"] and event.dispatch_recommended),
                    operator_id="operator-demo-01",
                    note=item["operator_note"],
                )
                mission = build_mission_message(event, decision)

                event_messages.append(event)
                if mission is not None:
                    mission_messages.append(mission)

                rows.append(
                    {
                        "event_id": event.event_id,
                        "location": event.location_label,
                        "score": event.risk_score,
                        "recommendation": event.recommendation_label,
                        "dispatch_recommended": event.dispatch_recommended,
                        "human_approved": decision.approved,
                        "mission": mission.mission_id if mission else "not_launched",
                        "payload_kit": mission.payload_kit if mission else "-",
                    }
                )

            summary_df = pd.DataFrame(rows)
            display(summary_df)
            """
        ),
        code_cell(
            """
            print("=== Event message example ===")
            print(json.dumps(asdict(event_messages[1]), ensure_ascii=False, indent=2))

            print("\\n=== Drone mission message example ===")
            print(json.dumps(asdict(mission_messages[0]), ensure_ascii=False, indent=2))
            """
        ),
        md_cell(
            """
            ## Interpretation

            이 데모에서 AI는 출동을 자동 결정하지 않는다. AI는 사건 메시지와 권고 단계를 생성하고, 출동 여부는 인간 관제자가 최종 승인한다.

            승인 이후 생성되는 드론 임무 메시지는 목적지, 우선순위, 통신망, 경로 정책, 탑재 키트까지 포함한다. 이후 `drone/` 노트북은 이 임무를 받아 지상 피해 기반 경로 계획, 실시간 이상 감지, 낙하산-충격 완화체-에어백 시퀀스로 연결되는 내부 안전 계층을 시연한다.
            """
        ),
    ]
    return notebook(cells)


def main():
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH.resolve())


if __name__ == "__main__":
    main()
