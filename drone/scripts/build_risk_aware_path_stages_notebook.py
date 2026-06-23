import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "RiskAware_Path_Stages_Colab.ipynb"


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


def build_notebook():
    cells = [
        md_cell(
            """
            # Risk-Aware Path Planning Stages Demo

            이 노트북은 안전 경로 계획의 설계 논리를 **단계별 코드 실험**으로 보여주기 위한 보조 데모이다.

            핵심 질문은 다음과 같다.

            1. 건물과 제한 구역만 피하면 충분한가?
            2. 건물에 너무 가까운 경로는 왜 불리한가?
            3. `건물 / 비건물` 이진 분류만으로 지상 위험을 설명할 수 있는가?
            4. 비상 추락을 고려한다면, 경로 평가는 어떻게 달라져야 하는가?

            따라서 본 노트북은 다음 5단계를 순차적으로 보여준다.

            - **Step 1**: 하드 제약만 사용하는 기본 경로
            - **Step 2**: 건물 이격 거리 패널티 추가
            - **Step 3**: geometry-only ground risk prior 추가
            - **Step 4**: semantic ground risk 개선안 추가
            - **Step 5**: 추락 영향 범위를 반영한 expected harm 기반 개선

            **중요:** 현재 메인 경로 계획 노트북은 대체로 **Step 3 수준**에 해당한다.
            본 노트북의 Step 4~5는 그 한계를 어떤 방향으로 보완할 수 있는지 보여주는 확장 데모이다.
            """
        ),
        code_cell(
            """
            import heapq
            import math

            import matplotlib.colors as mcolors
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from scipy.ndimage import binary_dilation, convolve, distance_transform_edt

            np.random.seed(7)
            plt.rcParams["figure.dpi"] = 130
            plt.rcParams["font.size"] = 10
            plt.rcParams["axes.titlesize"] = 12
            plt.rcParams["axes.labelsize"] = 10
            """
        ),
        md_cell(
            """
            ## 0. 합성 도심 환경 구성

            실제 메인 노트북은 3차원 STL 기반 경로 계획을 수행하지만,
            본 노트북은 **상단(top-view) 관점에서 단계별 설계 의사결정**을 보여주는 데 집중한다.

            아래 합성 환경에는 다음 요소를 포함하였다.

            - 건물 블록
            - 주행 도로와 교차로
            - 인도
            - 공원
            - 광장
            - 학교 주변 보호 구역(예시)
            """
        ),
        code_cell(
            """
            H, W = 80, 120

            building_mask = np.zeros((H, W), dtype=bool)

            def add_rect(mask, x0, y0, x1, y1):
                mask[y0:y1, x0:x1] = True

            # Main building masses
            add_rect(building_mask, 28, 18, 42, 62)
            add_rect(building_mask, 50, 8, 68, 46)
            add_rect(building_mask, 74, 28, 92, 72)

            road_mask = np.zeros((H, W), dtype=bool)
            road_mask[34:46, :] = True
            road_mask[:, 56:64] = True

            intersection_mask = np.zeros((H, W), dtype=bool)
            intersection_mask[32:48, 54:66] = True

            park_mask = np.zeros((H, W), dtype=bool)
            park_mask[8:24, 8:110] = True

            plaza_mask = np.zeros((H, W), dtype=bool)
            plaza_mask[58:74, 8:112] = True

            sidewalk_mask = np.zeros((H, W), dtype=bool)
            sidewalk_mask[30:34, :] = True
            sidewalk_mask[46:50, :] = True

            school_zone_mask = np.zeros((H, W), dtype=bool)
            school_zone_mask[10:28, 94:114] = True

            start = (6, 40)
            goal = (112, 40)

            semantic_preview = np.full((H, W), 0.18, dtype=float)
            semantic_preview[plaza_mask] = 0.24
            semantic_preview[park_mask] = 0.08
            semantic_preview[road_mask] = 0.34
            semantic_preview[sidewalk_mask] = 0.45
            semantic_preview[intersection_mask] = 0.72
            semantic_preview[school_zone_mask] = np.maximum(semantic_preview[school_zone_mask], 0.65)
            semantic_preview[building_mask] = 0.95
            """
        ),
        code_cell(
            """
            def plot_environment_overview():
                fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)

                # Left: semantic heat map
                im = axes[0].imshow(semantic_preview, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
                axes[0].scatter([start[0], goal[0]], [start[1], goal[1]], c=["lime", "cyan"], s=90, edgecolors="black")
                axes[0].text(start[0] + 2, start[1] + 2, "Start", color="lime", weight="bold")
                axes[0].text(goal[0] - 9, goal[1] + 2, "Goal", color="cyan", weight="bold")
                axes[0].set_title("합성 도심 환경의 예시적 지상 위험도")
                plt.colorbar(im, ax=axes[0], label="illustrative ground risk")

                # Right: class overlay
                class_map = np.zeros((H, W, 4), dtype=float)
                class_map[park_mask] = mcolors.to_rgba("#9fd18b")
                class_map[plaza_mask] = mcolors.to_rgba("#f4c27b")
                class_map[road_mask] = mcolors.to_rgba("#7f8c8d")
                class_map[sidewalk_mask] = mcolors.to_rgba("#bdc3c7")
                class_map[intersection_mask] = mcolors.to_rgba("#e74c3c")
                class_map[school_zone_mask] = mcolors.to_rgba("#f1c40f")
                class_map[building_mask] = mcolors.to_rgba("#2c3e50")

                axes[1].imshow(np.ones((H, W)), origin="lower", cmap="gray", vmin=0, vmax=1)
                axes[1].imshow(class_map, origin="lower")
                axes[1].scatter([start[0], goal[0]], [start[1], goal[1]], c=["lime", "cyan"], s=90, edgecolors="black")
                axes[1].set_title("환경 구성 요소 분해")

                legend_handles = [
                    mpatches.Patch(color="#2c3e50", label="건물"),
                    mpatches.Patch(color="#7f8c8d", label="도로"),
                    mpatches.Patch(color="#e74c3c", label="교차로"),
                    mpatches.Patch(color="#bdc3c7", label="인도"),
                    mpatches.Patch(color="#9fd18b", label="공원"),
                    mpatches.Patch(color="#f4c27b", label="광장"),
                    mpatches.Patch(color="#f1c40f", label="학교 주변"),
                ]
                axes[1].legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)

                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.show()

            plot_environment_overview()
            """
        ),
        md_cell(
            """
            ## 공통 도구 함수

            모든 단계는 동일한 기본 프레임 위에서 비교한다.

            - `safe_grid`: 건물에 safety buffer를 적용한 하드 제약
            - `dist_map`: 각 지점에서 가장 가까운 건물까지의 거리
            - `astar_plan`: 단계별 비용 함수를 반영한 2차원 경로 탐색
            - `path_metrics`: 경로 길이, 최소 이격 거리, 평균 위험도 등을 요약
            - `plot_stage_result`: 단계별 결과를 일관된 시각화 형식으로 표현
            """
        ),
        code_cell(
            """
            SAFETY_BUFFER = 3
            CLEARANCE_REF = 7.0

            STAGE_COLORS = {
                "stage1": "#1f77b4",
                "stage2": "#ff7f0e",
                "stage3": "#d62728",
                "stage4": "#7f3fbf",
                "stage5": "#17becf",
            }

            def neighbors8(x, y, width, height):
                for dx, dy in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        yield nx, ny, math.hypot(dx, dy)

            def heuristic(a, b):
                return math.hypot(a[0] - b[0], a[1] - b[1])

            def reconstruct_path(came_from, current):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            def astar_plan(start_xy, goal_xy, hard_forbidden, dist_map,
                           risk_map=None, clearance_weight=0.0, risk_weight=0.0,
                           clearance_ref=CLEARANCE_REF):
                height, width = hard_forbidden.shape
                open_heap = [(heuristic(start_xy, goal_xy), 0.0, start_xy)]
                came_from = {}
                g_score = {start_xy: 0.0}

                while open_heap:
                    _, current_g, current = heapq.heappop(open_heap)
                    if current == goal_xy:
                        return reconstruct_path(came_from, current)
                    if current_g > g_score.get(current, float("inf")):
                        continue

                    cx, cy = current
                    for nx, ny, move_cost in neighbors8(cx, cy, width, height):
                        if hard_forbidden[ny, nx]:
                            continue

                        clearance_penalty = 0.0
                        if clearance_weight > 0.0:
                            gap = max(0.0, clearance_ref - dist_map[ny, nx])
                            clearance_penalty = clearance_weight * (gap ** 2)

                        risk_penalty = 0.0
                        if risk_map is not None and risk_weight > 0.0:
                            risk_penalty = risk_weight * float(risk_map[ny, nx])

                        tentative_g = current_g + move_cost + clearance_penalty + risk_penalty
                        if tentative_g < g_score.get((nx, ny), float("inf")):
                            came_from[(nx, ny)] = current
                            g_score[(nx, ny)] = tentative_g
                            f_score = tentative_g + heuristic((nx, ny), goal_xy)
                            heapq.heappush(open_heap, (f_score, tentative_g, (nx, ny)))

                raise RuntimeError("No feasible path found in the staged demo.")

            def path_length(path):
                total = 0.0
                for a, b in zip(path[:-1], path[1:]):
                    total += math.hypot(a[0] - b[0], a[1] - b[1])
                return total

            def path_metrics(name, path, dist_map, raw_risk_map=None):
                xs = np.array([p[0] for p in path], dtype=int)
                ys = np.array([p[1] for p in path], dtype=int)
                data = {
                    "stage": name,
                    "length": round(path_length(path), 2),
                    "min_clearance": round(float(np.min(dist_map[ys, xs])), 2),
                    "mean_clearance": round(float(np.mean(dist_map[ys, xs])), 2),
                }
                if raw_risk_map is not None:
                    data["avg_risk_on_path"] = round(float(np.mean(raw_risk_map[ys, xs])), 3)
                    data["max_risk_on_path"] = round(float(np.max(raw_risk_map[ys, xs])), 3)
                return data

            def path_on_axes(ax, path, color, label, linewidth=2.6, alpha=1.0):
                ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=linewidth, alpha=alpha, label=label)
                ax.scatter([start[0], goal[0]], [start[1], goal[1]], c=["lime", "cyan"], s=75, edgecolors="black", zorder=5)

            def plot_stage_result(stage_title, stage_key, background, cmap, path,
                                  note_title, note_body, compare_path=None, compare_label=None,
                                  compare_color="#777777", vmin=None, vmax=None, cbar_label=None):
                fig, axes = plt.subplots(
                    1, 2,
                    figsize=(14, 5.6),
                    gridspec_kw={"width_ratios": [1.15, 1.0]},
                    constrained_layout=True,
                )

                im = axes[0].imshow(background, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                path_on_axes(axes[0], path, STAGE_COLORS[stage_key], stage_title)
                axes[0].set_title(stage_title)
                if cbar_label is not None:
                    plt.colorbar(im, ax=axes[0], label=cbar_label)

                axes[1].axis("off")
                axes[1].text(0.02, 0.95, note_title, transform=axes[1].transAxes, fontsize=13, weight="bold", va="top")
                axes[1].text(0.02, 0.84, note_body, transform=axes[1].transAxes, fontsize=10.5, va="top", linespacing=1.55)

                if compare_path is not None:
                    inset = axes[1].inset_axes([0.08, 0.10, 0.84, 0.46])
                    inset.imshow(background, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                    path_on_axes(inset, compare_path, compare_color, compare_label, linewidth=2.1, alpha=0.75)
                    path_on_axes(inset, path, STAGE_COLORS[stage_key], stage_title, linewidth=2.6)
                    inset.set_title("이전 단계 대비 경로 변화", fontsize=10)
                    inset.set_xticks([])
                    inset.set_yticks([])
                    inset.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)

                for ax in axes:
                    if hasattr(ax, "set_xticks"):
                        ax.set_xticks([])
                        ax.set_yticks([])

                plt.show()

            def print_stage_metrics(metric_dict):
                display(pd.DataFrame([metric_dict]))

            safe_grid = binary_dilation(building_mask, iterations=SAFETY_BUFFER)
            dist_map = distance_transform_edt(~building_mask)
            """
        ),
        md_cell(
            """
            ## Step 1. 하드 제약만 사용하는 기본 경로

            첫 단계는 가장 단순한 출발점이다.

            - 건물과 safety buffer만 피한다.
            - 그 외 자유 공간은 모두 같은 비용으로 간주한다.

            즉, **“충돌하지 않는 경로”**는 만들 수 있지만,
            **“상대적으로 덜 위험한 경로”**를 구분할 수는 없다.
            """
        ),
        code_cell(
            """
            path_stage1 = astar_plan(
                start, goal,
                hard_forbidden=safe_grid,
                dist_map=dist_map,
                risk_map=None,
                clearance_weight=0.0,
                risk_weight=0.0,
            )

            metrics_stage1 = path_metrics("stage1_hard_only", path_stage1, dist_map)

            plot_stage_result(
                stage_title="Step 1 - 하드 제약만 반영한 기본 경로",
                stage_key="stage1",
                background=safe_grid.astype(float),
                cmap="gray_r",
                path=path_stage1,
                note_title="관찰되는 문제",
                note_body=(
                    "- 충돌은 피하지만, 건물과 매우 가까운 구간이 남을 수 있다.\\n"
                    "- 자유 공간 전체를 동일하게 보기 때문에, 도로·교차로·공원 같은 지상 맥락을 구분하지 못한다.\\n"
                    "- 즉, 경로는 '가능(feasible)'하지만 아직 '위험 회피적(risk-aware)'이라고 보기 어렵다."
                ),
                cbar_label=None,
            )

            print_stage_metrics(metrics_stage1)
            """
        ),
        md_cell(
            """
            ## Step 2. 건물 이격 거리 패널티 추가

            두 번째 단계에서는 `dist_map`을 이용해 **건물과의 여유 거리**를 비용에 반영한다.

            - 건물에 가까운 구간일수록 비용 증가
            - 단순 충돌 회피를 넘어, 구조물에서 떨어진 경로 선호
            """
        ),
        code_cell(
            """
            path_stage2 = astar_plan(
                start, goal,
                hard_forbidden=safe_grid,
                dist_map=dist_map,
                risk_map=None,
                clearance_weight=0.75,
                risk_weight=0.0,
            )

            metrics_stage2 = path_metrics("stage2_clearance", path_stage2, dist_map)

            plot_stage_result(
                stage_title="Step 2 - 건물 이격 거리 패널티 추가",
                stage_key="stage2",
                background=dist_map,
                cmap="viridis",
                path=path_stage2,
                note_title="개선되는 점",
                note_body=(
                    "- 건물에 지나치게 가까운 경로가 불리해진다.\\n"
                    "- 최소 이격 거리와 평균 이격 거리가 개선될 수 있다.\\n"
                    "- 다만 지상 위험은 아직 고려하지 않으므로, 비건물 공간 내부의 차이는 여전히 반영되지 않는다."
                ),
                compare_path=path_stage1,
                compare_label="Step 1",
                compare_color=STAGE_COLORS["stage1"],
                cbar_label="nearest-building distance",
            )

            print_stage_metrics(metrics_stage2)
            """
        ),
        md_cell(
            """
            ## Step 3. geometry-only ground risk prior 추가

            세 번째 단계는 현재 메인 경로 계획 코드의 핵심 구조와 가장 유사한 단계이다.

            - building: 높은 위험도
            - non-building: 낮은 기본 위험도

            이 단계는 **건물 회피 + 단순화된 지상 위험 prior**를 결합한다.
            """
        ),
        code_cell(
            """
            geometry_risk_map = np.full((H, W), 0.15, dtype=float)
            geometry_risk_map[building_mask] = 0.95

            path_stage3 = astar_plan(
                start, goal,
                hard_forbidden=safe_grid,
                dist_map=dist_map,
                risk_map=geometry_risk_map,
                clearance_weight=0.75,
                risk_weight=5.0,
            )

            metrics_stage3 = path_metrics("stage3_geometry_only", path_stage3, dist_map, geometry_risk_map)

            plot_stage_result(
                stage_title="Step 3 - geometry-only ground-risk prior",
                stage_key="stage3",
                background=geometry_risk_map,
                cmap="YlOrRd",
                path=path_stage3,
                note_title="현재 메인 코드와 연결되는 단계",
                note_body=(
                    "- 현재 메인 경로 계획은 대체로 이 단계와 유사하다.\\n"
                    "- building / non-building 구분만 있으므로, 도로를 안전하다고 '판정'하는 것이 아니라 도로를 포함한 비건물 지면을 충분히 구분하지 못한다.\\n"
                    "- 즉, road·sidewalk·park·plaza가 모두 거의 같은 저위험 범주에 남는다."
                ),
                compare_path=path_stage2,
                compare_label="Step 2",
                compare_color=STAGE_COLORS["stage2"],
                vmin=0,
                vmax=1,
                cbar_label="geometry-only ground risk",
            )

            print_stage_metrics(metrics_stage3)
            """
        ),
        md_cell(
            """
            ## Step 4. semantic ground risk 개선안

            네 번째 단계에서는 비건물 지면 내부의 차이를 직접 반영한다.

            - road / sidewalk / intersection / park / plaza 구분
            - 같은 비건물이라도 의미가 다르면 위험도도 다르게 부여

            이 단계는 **“비건물 전체를 하나로 보는 한계”**를 보완하는 방향을 보여준다.
            """
        ),
        code_cell(
            """
            semantic_risk_map = np.full((H, W), 0.18, dtype=float)
            semantic_risk_map[plaza_mask] = 0.24
            semantic_risk_map[park_mask] = 0.08
            semantic_risk_map[road_mask] = 0.34
            semantic_risk_map[sidewalk_mask] = 0.45
            semantic_risk_map[intersection_mask] = 0.72
            semantic_risk_map[school_zone_mask] = np.maximum(semantic_risk_map[school_zone_mask], 0.65)
            semantic_risk_map[building_mask] = 0.95

            path_stage4 = astar_plan(
                start, goal,
                hard_forbidden=safe_grid,
                dist_map=dist_map,
                risk_map=semantic_risk_map,
                clearance_weight=0.75,
                risk_weight=5.0,
            )

            metrics_stage4 = path_metrics("stage4_semantic_risk", path_stage4, dist_map, semantic_risk_map)

            plot_stage_result(
                stage_title="Step 4 - semantic ground risk 개선안",
                stage_key="stage4",
                background=semantic_risk_map,
                cmap="YlOrRd",
                path=path_stage4,
                note_title="개선되는 점",
                note_body=(
                    "- 도로, 인도, 교차로, 공원, 광장 간의 차이가 경로 비용에 반영된다.\\n"
                    "- 같은 비건물 공간이라도 조건부 위험 차이를 표현할 수 있다.\\n"
                    "- 특히 교차로, 인도, 학교 주변처럼 더 민감한 영역을 우회하는 방향으로 경로가 재조정될 수 있다."
                ),
                compare_path=path_stage3,
                compare_label="Step 3",
                compare_color=STAGE_COLORS["stage3"],
                vmin=0,
                vmax=1,
                cbar_label="semantic ground risk",
            )

            print_stage_metrics(metrics_stage4)
            """
        ),
        md_cell(
            """
            ## Step 5. 추락 영향 범위를 반영한 expected harm 기반 개선

            다섯 번째 단계는 본 연구의 핵심 확장을 가장 직접적으로 보여준다.

            - 현재 위치 바로 아래 지점만 보는 것이 아니라,
            - **비상 추락 시 영향을 줄 수 있는 영역 전체**를 함께 본다.

            이를 위해 semantic risk map을 바로 쓰는 대신,
            추락 영향 반경을 반영한 `impact_risk_map`으로 변환한 뒤 다시 경로를 탐색한다.
            """
        ),
        code_cell(
            """
            altitude_m = 60.0
            horizontal_speed_mps = 12.0
            gravity = 9.81
            impact_margin = 4.0

            impact_radius = horizontal_speed_mps * math.sqrt(2.0 * altitude_m / gravity) + impact_margin
            impact_radius_cells = max(2, int(round(impact_radius / 6.0)))

            yy, xx = np.ogrid[-impact_radius_cells:impact_radius_cells+1, -impact_radius_cells:impact_radius_cells+1]
            kernel_mask = (xx * xx + yy * yy) <= impact_radius_cells * impact_radius_cells
            impact_kernel = kernel_mask.astype(float)
            impact_kernel /= impact_kernel.sum()

            impact_risk_map = convolve(semantic_risk_map, impact_kernel, mode="nearest")

            path_stage5 = astar_plan(
                start, goal,
                hard_forbidden=safe_grid,
                dist_map=dist_map,
                risk_map=impact_risk_map,
                clearance_weight=0.75,
                risk_weight=8.0,
            )

            metrics_stage5 = path_metrics("stage5_expected_harm", path_stage5, dist_map, impact_risk_map)

            plot_stage_result(
                stage_title="Step 5 - expected harm 기반 개선",
                stage_key="stage5",
                background=impact_risk_map,
                cmap="magma",
                path=path_stage5,
                note_title="핵심 확장",
                note_body=(
                    "- 추락 시 영향을 줄 수 있는 수평 범위를 포함해 경로를 평가한다.\\n"
                    "- 따라서 단순 '현재 위치 아래'의 위험이 아니라, 추락 footprint와 결합된 expected harm 관점이 반영된다.\\n"
                    "- 이는 본 연구가 단순 충돌 회피를 넘어 비상 추락 피해 저감 경로로 확장된다는 점을 가장 잘 보여준다."
                ),
                compare_path=path_stage4,
                compare_label="Step 4",
                compare_color=STAGE_COLORS["stage4"],
                cbar_label="expected harm proxy",
            )

            print_stage_metrics(metrics_stage5)
            """
        ),
        md_cell(
            """
            ## 단계별 비교 요약

            아래 표는 각 단계가 실제로 무엇을 개선하는지 비교하기 위한 요약값이다.
            """
        ),
        code_cell(
            """
            summary_df = pd.DataFrame([
                metrics_stage1,
                metrics_stage2,
                metrics_stage3,
                metrics_stage4,
                metrics_stage5,
            ])

            summary_df["delta_length_from_prev"] = summary_df["length"].diff().round(2)
            summary_df["delta_min_clearance_from_prev"] = summary_df["min_clearance"].diff().round(2)
            summary_df
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
            stages = ["S1", "S2", "S3", "S4", "S5"]

            axes[0].bar(stages, summary_df["length"], color=["#1f77b4", "#ff7f0e", "#d62728", "#7f3fbf", "#17becf"])
            axes[0].set_title("경로 길이 비교")
            axes[0].set_ylabel("path length")

            axes[1].bar(stages, summary_df["min_clearance"], color=["#1f77b4", "#ff7f0e", "#d62728", "#7f3fbf", "#17becf"])
            axes[1].set_title("최소 이격 거리 비교")
            axes[1].set_ylabel("minimum clearance")

            axes[2].bar(
                stages[2:],
                summary_df.loc[summary_df["stage"].isin(["stage3_geometry_only", "stage4_semantic_risk", "stage5_expected_harm"]), "avg_risk_on_path"],
                color=["#d62728", "#7f3fbf", "#17becf"],
            )
            axes[2].set_title("경로 평균 위험도 비교")
            axes[2].set_ylabel("average risk / harm proxy")

            for ax in axes:
                ax.grid(axis="y", alpha=0.25)

            plt.show()
            """
        ),
        md_cell(
            """
            ## 해석 포인트

            이 노트북의 의미는 다음과 같이 정리할 수 있다.

            1. **현재 구현 수준을 정직하게 보여준다.**
               - 현재 메인 경로 계획은 대체로 Step 3에 가깝다.
               - 즉, building avoidance + geometry-only ground-risk prior 수준이다.

            2. **왜 그것만으로는 부족한지 단계적으로 드러낸다.**
               - 건물과의 이격은 확보할 수 있어도,
               - 비건물 공간 내부의 의미적 차이와
               - 비상 추락 영향 범위를 충분히 반영하지 못한다.

            3. **어떤 방향으로 개선할 수 있는지 코드로 보여준다.**
               - semantic ground-risk map
               - impact-footprint-aware expected harm

            따라서 이 노트북은 단순한 개념 설명이 아니라,
            **문제 제기 -> 단계별 구현 -> 한계 노출 -> 개선 방향 제시**의 흐름을 실제 코드와 결과 시각화로 보여주는 보조 실험 노트북이다.
            """
        ),
    ]
    return notebook(cells)


def main():
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH.resolve())


if __name__ == "__main__":
    main()
