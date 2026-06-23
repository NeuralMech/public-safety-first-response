import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQ_NOTEBOOK = PROJECT_ROOT / "notebooks" / "EmergencySequencer_Colab.ipynb"
INTEGRATED_NOTEBOOK = PROJECT_ROOT / "notebooks" / "FullIntegratedDroneSafety_Colab.ipynb"


def lines(text: str):
    return [line + "\n" for line in dedent(text).strip("\n").splitlines()]


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
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


def build_sequencer_notebook():
    cells = [
        md_cell(
            """
            # Emergency Sequencer Final Demo

            This notebook imports the final `EmergencySequencer` implementation directly
            and runs three representative descent scenarios.

            - Normal-altitude descent with parachute and low-risk airbag flow
            - Low-altitude fast-track descent with secondary cushion deployment
            - High-risk descent where the airbag is withheld and cushion-only mitigation remains
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import pandas as pd

            CWD = Path.cwd()
            SRC_CANDIDATES = [
                CWD / "src",
                CWD / "drone" / "src",
                CWD.parent / "src",
                CWD.parent / "drone" / "src",
                CWD.parent.parent / "drone" / "src",
                CWD,
            ]
            for candidate in SRC_CANDIDATES:
                if (candidate / "emergency_sequencer.py").exists():
                    if str(candidate) not in sys.path:
                        sys.path.insert(0, str(candidate))
                    break
            else:
                raise FileNotFoundError("Could not locate drone src directory containing emergency_sequencer.py")

            from emergency_sequencer import (
                CargoState,
                DescentPhysics,
                EmergencySequencer,
                ImpactRiskAssessment,
            )
            """
        ),
        code_cell(
            """
            def run_scenario(name, altitude, vertical_speed, risk, radar_distance=None, steps=80):
                seq = EmergencySequencer()
                physics = DescentPhysics(
                    initial_altitude=altitude,
                    initial_vertical_speed=vertical_speed,
                    dt=0.1,
                )

                history = []
                ctx = physics.snapshot()
                ctx.radar_distance = radar_distance
                ctx.rotor_safe = None
                ctx.cargo_state = CargoState.LIQUID_FULL_HEAVY
                ctx.impact_risk = risk

                result = seq.start(ctx)
                history.append(
                    {
                        "step": 0,
                        "phase": result.phase.name,
                        "altitude": ctx.altitude_agl,
                        "vertical_speed": ctx.vertical_speed,
                        "tti": ctx.time_to_impact,
                        "risk": risk.impact_zone_label,
                        "secondary_cushion_deploy": bool(result.commands.get("secondary_cushion_deploy", False)),
                        "parachute_deploy": bool(result.commands.get("parachute_deploy", False)),
                        "airbag_prefill": bool(result.commands.get("airbag_prefill", False)),
                        "airbag_fire": bool(result.commands.get("airbag_fire", False)),
                        "log": result.log,
                    }
                )

                if result.commands.get("parachute_deploy"):
                    physics.deploy_chute()

                for step in range(1, steps + 1):
                    if physics.is_terminal or seq.is_terminal:
                        break

                    physics.step(mass_kg=130.0)
                    ctx = physics.snapshot()
                    ctx.radar_distance = min(ctx.altitude_agl, 10.0)
                    ctx.rotor_safe = None
                    ctx.cargo_state = CargoState.LIQUID_FULL_HEAVY
                    ctx.impact_risk = risk

                    result = seq.step(ctx)
                    if result.commands.get("parachute_deploy"):
                        physics.deploy_chute()

                    history.append(
                        {
                            "step": step,
                            "phase": result.phase.name,
                            "altitude": ctx.altitude_agl,
                            "vertical_speed": ctx.vertical_speed,
                            "tti": ctx.time_to_impact,
                            "risk": risk.impact_zone_label,
                            "secondary_cushion_deploy": bool(result.commands.get("secondary_cushion_deploy", False)),
                            "parachute_deploy": bool(result.commands.get("parachute_deploy", False)),
                            "airbag_prefill": bool(result.commands.get("airbag_prefill", False)),
                            "airbag_fire": bool(result.commands.get("airbag_fire", False)),
                            "log": result.log,
                        }
                    )

                df = pd.DataFrame(history)
                print(f"=== {name} ===")
                display(df.tail(12))
                return df
            """
        ),
        code_cell(
            """
            low_risk = ImpactRiskAssessment.low_risk(score=0.18, reason="demo low-risk")
            high_risk = ImpactRiskAssessment.high_risk(score=0.82, reason="demo high-risk")

            scenario_a = run_scenario(
                "Scenario A: normal altitude / low risk",
                altitude=80.0,
                vertical_speed=8.0,
                risk=low_risk,
            )

            scenario_b = run_scenario(
                "Scenario B: low altitude fast-track / low risk",
                altitude=4.5,
                vertical_speed=12.0,
                risk=low_risk,
            )

            scenario_c = run_scenario(
                "Scenario C: low altitude fast-track / high risk",
                altitude=4.5,
                vertical_speed=12.0,
                risk=high_risk,
            )
            """
        ),
    ]
    return notebook(cells)


def build_integrated_notebook():
    cells = [
        md_cell(
            """
            # Full Integrated Drone Safety Final Demo

            This notebook demonstrates the final integrated drone safety flow.

            - `PathTracker` estimates route deviation and cross-track error
            - `Step1SafetyDetector` detects anomalies and issues EmergencyCommit
            - `EmergencySequencer` executes parachute + secondary cushion + airbag harm-mitigation logic

            Notes:
            - Real-time risk gating and staged airbag deployment remain concept-level modules.
            - This notebook is a system-logic demonstration, not a hardware validation notebook.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import torch
            import torch.nn as nn

            CWD = Path.cwd()
            SRC_CANDIDATES = [
                CWD / "src",
                CWD / "drone" / "src",
                CWD.parent / "src",
                CWD.parent / "drone" / "src",
                CWD.parent.parent / "drone" / "src",
                CWD,
            ]
            for candidate in SRC_CANDIDATES:
                if (candidate / "emergency_sequencer.py").exists():
                    if str(candidate) not in sys.path:
                        sys.path.insert(0, str(candidate))
                    break
            else:
                raise FileNotFoundError("Could not locate drone src directory containing emergency_sequencer.py")

            from emergency_sequencer import (
                CargoState,
                DescentPhysics,
                EmergencySequencer,
                ImpactRiskAssessment,
            )
            from path_tracker import PathTracker
            from step1_safety_detector import (
                FEATURE_ORDER,
                Step1ModelArtifact,
                Step1SafetyDetector,
            )
            """
        ),
        code_cell(
            """
            class DemoAnomalyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self._dummy = nn.Parameter(torch.zeros(1))

                def forward(self, x):
                    return self.predict_proba(x)

                def predict_proba(self, x):
                    last = x[:, -1, :]
                    accel_mag = torch.abs(last[:, 0]) + torch.abs(last[:, 1]) + torch.abs(last[:, 2] - 9.8)
                    gyro_mag = torch.abs(last[:, 3]) + torch.abs(last[:, 4])
                    cte = torch.abs(last[:, 5])
                    score = 0.28 * accel_mag + 0.35 * gyro_mag + 0.22 * cte - 2.2
                    return torch.sigmoid(score).reshape(-1, 1)


            model = DemoAnomalyModel()
            artifact = Step1ModelArtifact.from_model(
                model=model,
                seq_length=10,
                feature_columns=FEATURE_ORDER,
                source="demo-final",
                strict_feature_columns=True,
            )
            detector = Step1SafetyDetector.from_artifact(artifact)
            """
        ),
        code_cell(
            """
            def build_demo_path(n_points=240):
                t = np.linspace(0.0, 1.0, n_points)
                x = 120.0 * t
                y = 6.0 * np.sin(2.0 * np.pi * t)
                z = 42.0 - 18.0 * t
                return np.column_stack([x, y, z]).astype(np.float32)


            planned_path = build_demo_path()
            tracker = PathTracker(planned_path, off_route_threshold_m=5.0, search_window=25, ema_alpha=0.3)
            planned_path[:3], planned_path[-3:]
            """
        ),
        code_cell(
            """
            def make_sensor_payload(step_idx, tracking_state, fault_onset_step):
                if step_idx < fault_onset_step:
                    accel_x = np.random.normal(0.0, 0.15)
                    accel_y = np.random.normal(0.0, 0.15)
                    accel_z = np.random.normal(9.8, 0.2)
                    gyro_x = np.random.normal(0.0, 0.05)
                    gyro_y = np.random.normal(0.0, 0.05)
                else:
                    severity = min(1.0, (step_idx - fault_onset_step) / 10.0)
                    accel_x = np.random.normal(3.0 * severity, 0.5)
                    accel_y = np.random.normal(-2.5 * severity, 0.5)
                    accel_z = np.random.normal(9.8 - 5.0 * severity, 0.6)
                    gyro_x = np.random.normal(4.5 * severity, 0.4)
                    gyro_y = np.random.normal(3.8 * severity, 0.4)

                return {
                    "accel_x": float(accel_x),
                    "accel_y": float(accel_y),
                    "accel_z": float(accel_z),
                    "gyro_x": float(gyro_x),
                    "gyro_y": float(gyro_y),
                    "cross_track_error": float(tracking_state.cross_track_error),
                }


            def make_risk_snapshot(current_pos):
                x = float(current_pos[0])
                if x < 75.0:
                    return ImpactRiskAssessment.low_risk(score=0.22, reason="demo low-risk corridor")
                return ImpactRiskAssessment.high_risk(score=0.78, reason="demo dense urban block")
            """
        ),
        code_cell(
            """
            def run_full_pipeline(total_steps=80, fault_onset_step=26):
                detector.reset()
                tracker.reset()

                seq = None
                physics = None
                rows = []

                for step in range(total_steps):
                    planned_idx = min(step * 2, len(planned_path) - 1)
                    current_pos = planned_path[planned_idx].copy()

                    if step >= fault_onset_step:
                        drift = min(10.0, 0.6 * (step - fault_onset_step + 1))
                        current_pos[1] += drift
                        current_pos[2] = max(4.5, current_pos[2] - 0.9 * (step - fault_onset_step + 1))

                    tracking = tracker.update(current_pos)
                    sensor = make_sensor_payload(step, tracking, fault_onset_step)
                    detection = detector.step(sensor)

                    seq_result = None
                    if detection.commit_step2 and seq is None:
                        seq = EmergencySequencer()
                        physics = DescentPhysics(
                            initial_altitude=max(4.0, float(current_pos[2])),
                            initial_vertical_speed=6.0,
                            dt=0.1,
                        )
                        ctx = physics.snapshot()
                        ctx.radar_distance = min(ctx.altitude_agl, 10.0)
                        ctx.rotor_safe = None
                        ctx.cargo_state = CargoState.LIQUID_FULL_HEAVY
                        ctx.impact_risk = make_risk_snapshot(current_pos)
                        seq_result = seq.start(ctx)
                        if seq_result.commands.get("parachute_deploy"):
                            physics.deploy_chute()

                    elif seq is not None and physics is not None and not seq.is_terminal and not physics.is_terminal:
                        physics.step(mass_kg=130.0)
                        ctx = physics.snapshot()
                        ctx.radar_distance = min(ctx.altitude_agl, 10.0)
                        ctx.rotor_safe = None
                        ctx.cargo_state = CargoState.LIQUID_FULL_HEAVY
                        ctx.impact_risk = make_risk_snapshot(current_pos)
                        seq_result = seq.step(ctx)
                        if seq_result.commands.get("parachute_deploy"):
                            physics.deploy_chute()

                    rows.append(
                        {
                            "step": step,
                            "planned_x": float(planned_path[planned_idx, 0]),
                            "planned_y": float(planned_path[planned_idx, 1]),
                            "planned_z": float(planned_path[planned_idx, 2]),
                            "actual_x": float(current_pos[0]),
                            "actual_y": float(current_pos[1]),
                            "actual_z": float(current_pos[2]),
                            "cross_track_error_m": float(tracking.cross_track_error),
                            "cross_track_error_filtered_m": float(tracking.cross_track_error_filtered),
                            "lstm_probability": float(detection.lstm_probability),
                            "state": detection.state.name,
                            "commit_step2": bool(detection.commit_step2),
                            "phase": (seq_result.phase.name if seq_result is not None else (seq.phase.name if seq else "IDLE")),
                            "secondary_cushion_deploy": bool(seq_result.commands.get("secondary_cushion_deploy", False)) if seq_result else False,
                            "parachute_deploy": bool(seq_result.commands.get("parachute_deploy", False)) if seq_result else False,
                            "airbag_prefill": bool(seq_result.commands.get("airbag_prefill", False)) if seq_result else False,
                            "airbag_fire": bool(seq_result.commands.get("airbag_fire", False)) if seq_result else False,
                            "impact_risk_label": (make_risk_snapshot(current_pos).impact_zone_label if seq is not None else "n/a"),
                            "message": detection.message,
                        }
                    )

                return pd.DataFrame(rows)
            """
        ),
        code_cell(
            """
            history_df = run_full_pipeline()
            display(history_df.tail(20))
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            axes[0].plot(history_df["step"], history_df["lstm_probability"], label="LSTM probability")
            axes[0].axhline(0.85, color="tab:red", linestyle="--", label="suspect threshold")
            axes[0].set_ylabel("Probability")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(history_df["step"], history_df["cross_track_error_m"], label="CTE raw")
            axes[1].plot(history_df["step"], history_df["cross_track_error_filtered_m"], label="CTE filtered")
            axes[1].set_ylabel("Cross-track error [m]")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            phase_codes = {name: idx for idx, name in enumerate(pd.unique(history_df["phase"]))}
            axes[2].step(history_df["step"], history_df["phase"].map(phase_codes), where="post")
            axes[2].set_yticks(list(phase_codes.values()))
            axes[2].set_yticklabels(list(phase_codes.keys()))
            axes[2].set_ylabel("Emergency phase")
            axes[2].set_xlabel("Step")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            """
        ),
    ]
    return notebook(cells)


def main():
    SEQ_NOTEBOOK.write_text(
        json.dumps(build_sequencer_notebook(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    INTEGRATED_NOTEBOOK.write_text(
        json.dumps(build_integrated_notebook(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {SEQ_NOTEBOOK.name}")
    print(f"Wrote {INTEGRATED_NOTEBOOK.name}")


if __name__ == "__main__":
    main()
