import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_CANDIDATES = sorted(NOTEBOOK_DIR.glob("*v2.ipynb"))
if not NOTEBOOK_CANDIDATES:
    raise FileNotFoundError("Could not find the target v2 notebook in the notebooks folder.")
NOTEBOOK_PATH = NOTEBOOK_CANDIDATES[0]


def _lines(text: str):
    return [line + "\n" for line in text.rstrip("\n").splitlines()]


def _set_cell_source(nb, idx, text):
    nb["cells"][idx]["source"] = _lines(dedent(text).strip("\n"))


def _make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(dedent(text).strip("\n")),
    }


def _make_code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(dedent(text).strip("\n")),
    }


def _drop_cells_with_marker(nb, marker: str):
    filtered = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if marker in src:
            continue
        filtered.append(cell)
    nb["cells"] = filtered


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    _set_cell_source(
        nb,
        2,
        """
        !pip install trimesh
        !pip install ipympl
        !pip install kaleido

        import numpy as np
        import trimesh
        import heapq
        import math
        import time
        import scipy.ndimage as ndimage
        from scipy.interpolate import splprep, splev
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
        import matplotlib.pyplot as plt
        from pathlib import Path

        try:
            from google.colab import files
            IN_COLAB = True
        except ImportError:
            files = None
            IN_COLAB = False

        print(f"Google Colab detected: {IN_COLAB}")
        """,
    )

    _set_cell_source(
        nb,
        3,
        """
### 1-1. Hard safety margin and simplified ground-risk prior
This notebook uses **meter-based physical parameters** instead of hard-coded voxel counts.
The same configuration also includes the **geometry-only ground-risk prior** and the
**emergency impact footprint** parameters used by the risk-aware planner.
        """,
    )

    _set_cell_source(
        nb,
        4,
        """
# Safety-margin components (roughly aligned with the DJI T70P class)
DRONE_RADIUS_M          = 0.8
POSITION_ERROR_M        = 0.1
TRACKING_ERROR_M        = 0.6
WIND_MARGIN_M           = 1.2

SAFETY_MARGIN_BASE_M = (
    DRONE_RADIUS_M
    + POSITION_ERROR_M
    + TRACKING_ERROR_M
    + WIND_MARGIN_M
)
SAFETY_MARGIN_M = SAFETY_MARGIN_BASE_M

# Ground-risk / impact-footprint parameters
GRAVITY_MPS2                  = 9.81
DEFAULT_HORIZONTAL_SPEED_MPS  = 12.0
IMPACT_MARGIN_M               = 4.0
GROUND_RISK_WEIGHT            = 12.0
NON_BUILDING_BASE_RISK        = 0.15
BUILDING_RISK                 = 0.85

print(f"Safety margin total: {SAFETY_MARGIN_M:.2f} m")
print(f"Risk-aware planner weight: lambda_risk = {GROUND_RISK_WEIGHT:.2f}")
        """,
    )

    _set_cell_source(
        nb,
        6,
        """
print("1. STL 입력 준비")

def resolve_local_stl():
    candidates = [
        Path("sejong.stl"),
        Path("../assets/sejong.stl"),
        Path("assets/sejong.stl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

if IN_COLAB:
    print("   -> Google Colab 환경이 감지되어 업로드 창을 엽니다.")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No STL file was uploaded.")
    file_name = next(iter(uploaded.keys()))
else:
    print("   -> 로컬/일반 Jupyter 환경으로 판단하여 기본 STL 경로를 탐색합니다.")
    local_stl = resolve_local_stl()
    if local_stl is None:
        raise FileNotFoundError(
            "No local STL file was found. Place an STL file next to the notebook or in ../assets/."
        )
    file_name = str(local_stl)
    print(f"   -> 로컬 STL 사용: {file_name}")

print(f"\\n2. '{file_name}' 로드 및 3D 복셀화 진행 중...")
t0 = time.time()
mesh = trimesh.load(file_name)
max_extent = max(mesh.extents)

# 해상도 향상: 150 -> 200 (복셀화 오차를 줄임)
# auto_pitch가 작아지면 SAFETY_MARGIN을 복셀로 변환할 때 여유 해상도가 늘어남
RESOLUTION = 200
auto_pitch = max_extent / RESOLUTION
voxels = mesh.voxelized(pitch=auto_pitch)
original_grid = voxels.matrix.astype(int)
voxel_origin = mesh.bounds[0]

# 복셀화 오차는 auto_pitch의 절반
VOXELIZATION_ERROR_M = auto_pitch / 2.0
SAFETY_MARGIN_M = SAFETY_MARGIN_BASE_M + VOXELIZATION_ERROR_M

# 미터 마진을 복셀 수로 역산 (올림하여 보수적으로 잡음)
BUFFER_VOXELS = int(math.ceil(SAFETY_MARGIN_M / auto_pitch))

print(f"   -> 복셀화 완료! (소요 시간: {time.time()-t0:.2f}초)")
print(f"      맵 크기: {original_grid.shape}")
print(f"      auto_pitch: {auto_pitch:.3f} m")
print(f"      복셀화 오차: ±{VOXELIZATION_ERROR_M:.3f} m")
print(f"      최종 SAFETY_MARGIN: {SAFETY_MARGIN_M:.2f} m")
print(f"      BUFFER_VOXELS (역산): {BUFFER_VOXELS}")
        """,
    )

    _set_cell_source(
        nb,
        7,
        """
### 1-3. Hard-boundary preprocessing and ground-risk representation
This stage creates three maps.
- `safe_grid`: hard no-fly mask after applying the safety buffer
- `dist_map`: distance field used for wall-clearance penalties
- `ground_risk_map` / `impact_risk_stack`: XY prior and height-dependent
  **emergency impact risk** where `R_emg = K * rho_ground`
        """,
    )

    _set_cell_source(
        nb,
        8,
        """
print("Running safety preprocessing...")
t1 = time.time()

safe_grid = ndimage.binary_dilation(original_grid, iterations=BUFFER_VOXELS)
dist_map = ndimage.distance_transform_edt(1 - original_grid)

shape_x, shape_y, shape_z = original_grid.shape

ground_risk_map = np.full((shape_x, shape_y), NON_BUILDING_BASE_RISK, dtype=np.float32)

occupied_columns = np.any(original_grid > 0, axis=2)
bottom_z = np.argmax(original_grid > 0, axis=2)
top_z = shape_z - 1 - np.argmax((original_grid > 0)[:, :, ::-1], axis=2)
column_height_vox = np.where(occupied_columns, top_z - bottom_z + 1, 0)
column_height_m = column_height_vox * auto_pitch

# Treat only sufficiently tall vertical structures as building footprints so that
# extremely thin ground-like mesh artifacts do not dominate the risk map.
MIN_STRUCTURE_HEIGHT_M = max(10.0, 2.5 * auto_pitch)
MIN_STRUCTURE_HEIGHT_VOX = max(3, int(math.ceil(MIN_STRUCTURE_HEIGHT_M / auto_pitch)))

building_footprint = occupied_columns & (column_height_vox >= MIN_STRUCTURE_HEIGHT_VOX)
thin_surface_mask = occupied_columns & ~building_footprint

# Approximate roof height for roof-relative clearance evaluation.
roof_top_vox = np.where(occupied_columns, top_z + 1, 0)
roof_height_map_m = voxel_origin[2] + (roof_top_vox * auto_pitch)

ground_risk_map[building_footprint] = BUILDING_RISK

# Current version uses a geometry-only risk prior.
# Roads, sidewalks, plazas, parks, and other non-building surfaces are not
# semantically separated here; they all share the same non-building base risk.
ground_risk_map = np.clip(ground_risk_map, 0.0, 1.0)

def make_disk_kernel(radius_vox):
    yy, xx = np.ogrid[-radius_vox:radius_vox+1, -radius_vox:radius_vox+1]
    mask = (xx * xx + yy * yy) <= (radius_vox * radius_vox)
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel

def footprint_radius_m(height_m,
                       horizontal_speed_mps=DEFAULT_HORIZONTAL_SPEED_MPS,
                       margin_m=IMPACT_MARGIN_M):
    fall_time = math.sqrt(max(2.0 * max(height_m, 0.0), 0.0) / GRAVITY_MPS2)
    return horizontal_speed_mps * fall_time + margin_m

def build_impact_risk_stack(ground_risk_2d, pitch, n_z):
    impact_stack = np.zeros((shape_x, shape_y, n_z), dtype=np.float32)
    conv_cache = {}
    radius_schedule = []
    for nz in range(n_z):
        height_m = max(0.0, (nz + 1) * pitch)
        radius_m = footprint_radius_m(height_m)
        radius_vox = max(1, int(math.ceil(radius_m / pitch)))
        radius_schedule.append(radius_vox)
        if radius_vox not in conv_cache:
            kernel = make_disk_kernel(radius_vox)
            conv_cache[radius_vox] = ndimage.convolve(
                ground_risk_2d, kernel, mode="nearest"
            )
        impact_stack[:, :, nz] = conv_cache[radius_vox]
    return impact_stack, radius_schedule

impact_risk_stack, footprint_radius_schedule = build_impact_risk_stack(
    ground_risk_map, auto_pitch, shape_z
)

t_prep = time.time() - t1
print(f"Preprocessing complete: {t_prep:.2f} s")
print(f"  safe_grid shape          : {safe_grid.shape}")
print(f"  ground risk range        : {ground_risk_map.min():.2f} ~ {ground_risk_map.max():.2f}")
print(f"  impact radius vox range  : {min(footprint_radius_schedule)} ~ {max(footprint_radius_schedule)}")
print(f"  building footprint ratio : {building_footprint.mean():.3f}")
print(f"  thin surface ratio       : {thin_surface_mask.mean():.3f}")
print(f"  structure cutoff         : {MIN_STRUCTURE_HEIGHT_M:.2f} m ({MIN_STRUCTURE_HEIGHT_VOX} vox)")

mid_z = shape_z // 2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(ground_risk_map.T, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
axes[0].set_title("Geometry-only Ground-Risk Prior (building vs non-building)")
axes[0].set_xlabel("X voxel")
axes[0].set_ylabel("Y voxel")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(impact_risk_stack[:, :, mid_z].T, origin="lower", cmap="magma", vmin=0, vmax=1)
axes[1].set_title(
    f"Impact Risk Slice @ z={mid_z} (radius~{footprint_radius_schedule[mid_z]} vox)"
)
axes[1].set_xlabel("X voxel")
axes[1].set_ylabel("Y voxel")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.show()
        """,
    )

    _set_cell_source(
        nb,
        15,
        """
## Cell 3 - Improved A* planner (hard boundary + soft heuristics + ground risk)
`step_cost` is composed of the following terms.
1. Euclidean move cost
2. Low-altitude suppression term (`altitude_penalty`)
3. Wall-clearance penalty (`wall_penalty`)
4. **Emergency impact risk penalty** `lambda_risk * R_emg(n)`

Here `R_emg(n)` measures the expected ground risk if an emergency impact occurs at node `n`,
using the current impact footprint and the offline ground-risk prior.
        """,
    )

    _set_cell_source(
        nb,
        16,
        """
ALTITUDE_WEIGHT = 10.0
WALL_WEIGHT = 20.0
SOFT_WALL_DISTANCE_M = max(SAFETY_MARGIN_M, 2.0)
SAFE_WALL_DISTANCE = max(1, int(math.ceil(SOFT_WALL_DISTANCE_M / auto_pitch)))
H_WEIGHT = 2.0

print(f"Soft wall distance threshold: {SOFT_WALL_DISTANCE_M:.2f} m ({SAFE_WALL_DISTANCE} voxels)")
print(f"Ground risk weight: {GROUND_RISK_WEIGHT:.2f}")


def heuristic_uam_cruise(a, b, cruise_z):
    dist_2d = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    if dist_2d < 15.0:
        return math.sqrt(dist_2d**2 + (a[2]-b[2])**2)
    climb_cost = max(0, cruise_z - a[2])
    descend_cost = max(0, cruise_z - b[2])
    return dist_2d + climb_cost + descend_cost


def a_star_3d_safe_cruise(
    safe_grid_hard,
    dist_map_soft,
    impact_risk_stack_,
    start,
    goal,
    alt_weight,
    wall_weight,
    safe_wall_dist,
    ground_risk_weight,
):
    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]

    shape_x, shape_y, shape_z = safe_grid_hard.shape
    CRUISE_Z = shape_z + 2

    gscore = {start: 0}
    fscore = {start: heuristic_uam_cruise(start, goal, CRUISE_Z) * H_WEIGHT}
    came_from = {}
    close_set = set()
    open_set_hash = {start}
    oheap = [(fscore[start], start)]

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current in close_set:
            continue
        open_set_hash.discard(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        close_set.add(current)

        for dx, dy, dz in neighbors:
            nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz
            neighbor = (nx, ny, nz)

            if not (0 <= nx < shape_x and 0 <= ny < shape_y and nz >= 0):
                continue
            if neighbor in close_set:
                continue

            if nz < shape_z:
                if safe_grid_hard[nx, ny, nz] == 1:
                    continue
                wall_dist = dist_map_soft[nx, ny, nz]
                altitude_penalty = alt_weight / (nz + 1)
                impact_risk = float(impact_risk_stack_[nx, ny, nz])
            else:
                wall_dist = safe_wall_dist
                altitude_penalty = alt_weight / (shape_z + 1)
                impact_risk = float(impact_risk_stack_[nx, ny, -1])

            wall_penalty = (
                wall_weight / (wall_dist + 0.1)
                if wall_dist < safe_wall_dist else 0.0
            )
            risk_penalty = ground_risk_weight * impact_risk

            step_cost = (
                math.sqrt(dx * dx + dy * dy + dz * dz)
                + altitude_penalty
                + wall_penalty
                + risk_penalty
            )
            tentative_g = gscore[current] + step_cost

            if tentative_g < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + (
                    heuristic_uam_cruise(neighbor, goal, CRUISE_Z) * H_WEIGHT
                )
                if neighbor not in open_set_hash:
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    open_set_hash.add(neighbor)
    return None


print("Risk-aware A* function ready.")
        """,
    )

    _set_cell_source(
        nb,
        17,
        """
## Cell 4 - B-spline smoothing, hard-boundary verification, and impact-risk profile
After smoothing, the notebook:
1. checks whether the curve violates `safe_grid`
2. extracts the final-path `impact_risk_profile`
3. summarizes the planner's offline `ground_risk_map` prior visually
        """,
    )

    _set_cell_source(
        nb,
        18,
        """
def check_path_violations(path_real_coords, safe_grid_hard, voxel_origin_, pitch):
    voxel_indices = np.round((path_real_coords - voxel_origin_) / pitch).astype(int)
    shape = safe_grid_hard.shape
    violations = np.zeros(len(path_real_coords), dtype=bool)

    for i, (x, y, z) in enumerate(voxel_indices):
        if not (0 <= x < shape[0] and 0 <= y < shape[1]):
            violations[i] = True
            continue
        if z < 0:
            violations[i] = True
            continue
        if z >= shape[2]:
            continue
        if safe_grid_hard[x, y, z] == 1:
            violations[i] = True
    return violations


def find_violation_segments(violation_mask):
    segments = []
    in_segment = False
    seg_start = 0
    for i, v in enumerate(violation_mask):
        if v and not in_segment:
            seg_start = i
            in_segment = True
        elif not v and in_segment:
            segments.append((seg_start, i - 1))
            in_segment = False
    if in_segment:
        segments.append((seg_start, len(violation_mask) - 1))
    return segments


def smooth_and_verify(discrete_path_voxels, safe_grid_hard, voxel_origin_, pitch,
                      n_samples=500, max_iterations=3, verbose=True):
    path_np = np.array(discrete_path_voxels)
    discrete_real = voxel_origin_ + (path_np * pitch)

    diffs = np.sum(np.abs(np.diff(discrete_real, axis=0)), axis=1)
    keep = np.insert(diffs > 0, 0, True)
    filtered = discrete_real[keep]

    if len(filtered) <= 3:
        return filtered, {
            "iterations": [],
            "final_violations": 0,
            "fallback_used": True,
            "warning": "Path is too short for smoothing; returned the discrete path.",
        }

    report = {"iterations": [], "final_violations": 0, "fallback_used": False}

    seg_lengths = np.linalg.norm(np.diff(filtered, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    if cumulative[-1] == 0:
        return filtered, {
            "iterations": [],
            "final_violations": 0,
            "fallback_used": True,
            "warning": "Path length is zero after duplicate filtering; returned the discrete path.",
        }

    u_param = cumulative / cumulative[-1]
    spline_degree = min(3, len(filtered) - 1)

    # Use the full discrete path as the spline reference so the curve remains
    # close to the A* result while still smoothing out sharp turns.
    tck, _ = splprep(
        [filtered[:, 0], filtered[:, 1], filtered[:, 2]],
        u=u_param,
        s=1.5,
        k=spline_degree,
    )
    u_fine = np.linspace(0, 1, n_samples)
    smoothed = np.vstack(splev(u_fine, tck)).T

    for iteration in range(max_iterations):
        violations = check_path_violations(smoothed, safe_grid_hard, voxel_origin_, pitch)
        n_violations = int(violations.sum())
        report["iterations"].append({
            "iter": iteration,
            "violations": n_violations,
            "ratio": n_violations / n_samples,
        })
        if verbose:
            print(
                f"   [iter {iteration}] safety violations: "
                f"{n_violations}/{n_samples} ({100 * n_violations / n_samples:.1f}%)"
            )

        if n_violations == 0:
            report["final_violations"] = 0
            return smoothed, report

        segments = find_violation_segments(violations)
        smoothed_new = smoothed.copy()

        for seg_start, seg_end in segments:
            seg_u_start = seg_start / n_samples
            seg_u_end = (seg_end + 1) / n_samples
            disc_i_start = int(seg_u_start * (len(filtered) - 1))
            disc_i_end = min(int(seg_u_end * (len(filtered) - 1)) + 1, len(filtered))
            if disc_i_end - disc_i_start < 2:
                continue
            segment_points = filtered[disc_i_start:disc_i_end]
            n_segment = seg_end - seg_start + 1
            t_vals = np.linspace(0, 1, n_segment)
            interp_indices = (t_vals * (len(segment_points) - 1)).astype(int)
            smoothed_new[seg_start:seg_end + 1] = segment_points[interp_indices]

        smoothed = smoothed_new

    final_violations = check_path_violations(smoothed, safe_grid_hard, voxel_origin_, pitch)
    n_final = int(final_violations.sum())
    if n_final > 0:
        report["pre_fallback_violations"] = n_final
        report["final_violations"] = 0
        report["fallback_used"] = True
        report["fallback_reason"] = "Smoothed path still violated safety bounds, so the discrete path was restored."
        if verbose:
            print(f"   [fallback] final violations {n_final} -> revert to discrete path")
        return filtered, report

    report["final_violations"] = 0
    return smoothed, report


def sample_path_impact_risk(path_real_coords, impact_risk_stack_, voxel_origin_, pitch):
    voxel_indices = np.round((path_real_coords - voxel_origin_) / pitch).astype(int)
    shape = impact_risk_stack_.shape
    risks = []

    for x, y, z in voxel_indices:
        if not (0 <= x < shape[0] and 0 <= y < shape[1]):
            risks.append(1.0)
            continue
        if z < 0:
            risks.append(float(ground_risk_map[x, y]))
        elif z >= shape[2]:
            risks.append(float(impact_risk_stack_[x, y, -1]))
        else:
            risks.append(float(impact_risk_stack_[x, y, z]))
    return np.array(risks, dtype=np.float32)


def sample_roof_clearance(path_real_coords,
                          roof_height_map_m_,
                          building_footprint_,
                          voxel_origin_,
                          pitch):
    voxel_indices = np.round((path_real_coords - voxel_origin_) / pitch).astype(int)
    shape = building_footprint_.shape
    roof_clearance = []
    over_building = []

    for i, (x, y, _) in enumerate(voxel_indices):
        if not (0 <= x < shape[0] and 0 <= y < shape[1]):
            roof_clearance.append(np.nan)
            over_building.append(False)
            continue

        if building_footprint_[x, y]:
            roof_clearance.append(float(path_real_coords[i, 2] - roof_height_map_m_[x, y]))
            over_building.append(True)
        else:
            roof_clearance.append(np.nan)
            over_building.append(False)

    return np.array(roof_clearance, dtype=np.float32), np.array(over_building, dtype=bool)


# NOTE:
# In the final project description, `ground_risk_map` is used as an offline
# planning prior rather than a real-time safety authority.
# This helper only samples that offline prior for visualization/debugging.
def sample_offline_risk_prior(current_pos_real,
                              impact_risk_stack_,
                              voxel_origin_,
                              pitch):
    x, y, z = np.round((np.array(current_pos_real) - voxel_origin_) / pitch).astype(int)
    shape = impact_risk_stack_.shape

    if not (0 <= x < shape[0] and 0 <= y < shape[1]):
        score = 1.0
        reason = "Out of planner bounds; clamped to maximum geometry-only prior risk"
    elif z < 0:
        score = float(ground_risk_map[x, y])
        reason = f"Geometry-only ground prior score={score:.3f}"
    elif z >= shape[2]:
        score = float(impact_risk_stack_[x, y, -1])
        reason = f"Top-slice simplified R_emg prior={score:.3f}"
    else:
        score = float(impact_risk_stack_[x, y, z])
        reason = f"Simplified offline R_emg prior={score:.3f}"

    return {
        "impact_risk_score": score,
        "risk_band": (
            "low" if score < 0.33 else
            "medium" if score < 0.66 else
            "high"
        ),
        "risk_reason": reason,
    }


print("Path verification and offline risk-prior sampling helpers are ready.")
        """,
    )

    _set_cell_source(
        nb,
        19,
        """
## Cell 5 - Run the proposed planner (risk-aware final route)
This stage generates the route, extracts the final `impact_risk_profile`,
and packages the planner output into `planner_result`.
        """,
    )

    _set_cell_source(
        nb,
        20,
        """
print("1. Project the start/goal coordinates into safe_grid")
start_raw = real_to_voxel_index(USER_START_COORD)
goal_raw = real_to_voxel_index(USER_GOAL_COORD)
projection_search_radius = max(25, BUFFER_VOXELS * 6)

start_safe, start_proj_m = find_nearest_empty_with_projection(
    safe_grid, start_raw, max_radius=projection_search_radius
)
goal_safe, goal_proj_m = find_nearest_empty_with_projection(
    safe_grid, goal_raw, max_radius=projection_search_radius
)

print(f"   start projection distance: {start_proj_m:.2f} m")
print(f"   goal projection distance : {goal_proj_m:.2f} m")

if not np.isfinite(start_proj_m) or not np.isfinite(goal_proj_m):
    raise RuntimeError(
        "Could not find a valid projected start/goal inside safe_grid. "
        "Check the STL coverage or the requested coordinates."
    )
if safe_grid[start_safe] == 1 or safe_grid[goal_safe] == 1:
    raise RuntimeError("Projected start/goal is still inside the safety buffer.")

if start_proj_m > SAFETY_MARGIN_M * 3 or goal_proj_m > SAFETY_MARGIN_M * 3:
    print("   [warning] Projection distance exceeds 3x the safety margin.")
    print("             This usually means the requested endpoint is in a very risky region.")

print("\\n2. Run the improved A* search (safe_grid + simplified offline ground-risk prior)")
t0 = time.time()
discrete_path = a_star_3d_safe_cruise(
    safe_grid,
    dist_map,
    impact_risk_stack,
    start_safe,
    goal_safe,
    ALTITUDE_WEIGHT,
    WALL_WEIGHT,
    SAFE_WALL_DISTANCE,
    GROUND_RISK_WEIGHT,
)
t_astar = time.time() - t0
print(f"   search time: {t_astar:.2f} s")

if discrete_path is None:
    raise RuntimeError("No route found under the safe_grid + simplified offline ground-risk prior.")
print(f"   discrete path length: {len(discrete_path)} waypoints")

print("\\n3. Apply B-spline smoothing, hard verification, and risk profiling")
t0 = time.time()
smoothed_path_real, smooth_report = smooth_and_verify(
    discrete_path, safe_grid, voxel_origin, auto_pitch,
    n_samples=500, max_iterations=3, verbose=True
)
t_smooth = time.time() - t0
print(f"   smoothing time: {t_smooth:.2f} s")
print(f"   final violations: {smooth_report['final_violations']}/500")
if smooth_report.get("fallback_used"):
    print("   [info] Smoothing could not satisfy all safety constraints; fallback route was used.")
    if smooth_report.get("fallback_reason"):
        print(f"          reason: {smooth_report['fallback_reason']}")

discrete_path_real = voxel_origin + (np.array(discrete_path) * auto_pitch)
impact_risk_profile = sample_path_impact_risk(
    smoothed_path_real, impact_risk_stack, voxel_origin, auto_pitch
)

risk_summary = {
    "avg_impact_risk": float(np.mean(impact_risk_profile)),
}
print(f"   average impact risk: {risk_summary['avg_impact_risk']:.3f}")
print("   risk prior note    : buildings vs non-building surfaces only (geometry-only)")
print("   footprint note     : circular uniform impact kernel (1st-order approximation)")

planner_result = {
    "start_safe": start_safe,
    "goal_safe": goal_safe,
    "discrete_path": discrete_path,
    "discrete_path_real": discrete_path_real,
    "smoothed_path_real": smoothed_path_real,
    "smooth_report": smooth_report,
    "impact_risk_profile": impact_risk_profile,
    "risk_summary": risk_summary,
    "ground_risk_map": ground_risk_map,
    "impact_risk_stack": impact_risk_stack,
    "voxel_origin": voxel_origin,
    "auto_pitch": auto_pitch,
    "risk_model_note": (
        "Geometry-only prior: buildings vs non-building surfaces only; "
        "roads and other open-ground classes are not semantically separated."
    ),
    "footprint_model_note": (
        "Simplified circular impact kernel based on horizontal-speed and "
        "fall-time approximation."
    ),
}
        """,
    )

    _set_cell_source(
        nb,
        22,
        """

        def get_metrics_final_trajectory(
            path_real,
            safe_grid_hard,
            impact_risk_stack_,
            roof_height_map_m_,
            building_footprint_,
            voxel_origin_,
            pitch,
        ):
            if path_real is None or len(path_real) < 2:
                return None

            length = float(np.sum(np.linalg.norm(np.diff(path_real, axis=0), axis=1)))
            violations = check_path_violations(
                path_real, safe_grid_hard, voxel_origin_, pitch
            )
            impact_profile = sample_path_impact_risk(
                path_real, impact_risk_stack_, voxel_origin_, pitch
            )
            roof_clearance_profile, over_building_mask = sample_roof_clearance(
                path_real, roof_height_map_m_, building_footprint_, voxel_origin_, pitch
            )

            if np.any(over_building_mask):
                roof_vals = roof_clearance_profile[over_building_mask]
                min_roof_clearance_m = float(np.min(roof_vals))
            else:
                min_roof_clearance_m = float("nan")

            return {
                "hard_safety_pass": bool(np.sum(violations) == 0),
                "hard_safety_violations": int(np.sum(violations)),
                "length_m": length,
                "min_roof_clearance_m": min_roof_clearance_m,
                "over_building_fraction": float(np.mean(over_building_mask)),
                "avg_impact_risk": float(np.mean(impact_profile)),
                "n_points": len(path_real),
            }


        m_prop = get_metrics_final_trajectory(
            smoothed_path_real,
            safe_grid,
            impact_risk_stack,
            roof_height_map_m,
            building_footprint,
            voxel_origin,
            auto_pitch,
        )

        print("=== Proposed Algorithm Summary (final trajectory basis) ===")
        roof_clearance_text = (
            f"{m_prop['min_roof_clearance_m']:.2f} m"
            if not np.isnan(m_prop['min_roof_clearance_m'])
            else "N/A (no building overflight)"
        )
        hard_safety_text = (
            "PASS"
            if m_prop["hard_safety_pass"]
            else f"FAIL ({m_prop['hard_safety_violations']} violations)"
        )
        print(f"  Hard safety         : {hard_safety_text}")
        print(f"  Path length         : {m_prop['length_m']:.2f} m")
        print(f"  Min roof clearance  : {roof_clearance_text}")
        print(f"  Avg impact risk     : {m_prop['avg_impact_risk']:.3f}")

        """,
    )

    _set_cell_source(
        nb,
        24,
        """
        fig = go.Figure(data=[base_building_trace, base_edge_trace])

        real_start = voxel_to_real(start_safe)
        real_goal  = voxel_to_real(goal_safe)
        final_path_label = (
            'Smoothed Trajectory (verified + risk-aware)'
            if not smooth_report.get("fallback_used")
            else 'Verified Final Trajectory (fallback discrete path)'
        )
        smoothing_state = (
            'smoothed'
            if not smooth_report.get("fallback_used")
            else 'fallback to discrete path'
        )

        fig.add_trace(go.Scatter3d(
            x=discrete_path_real[:, 0],
            y=discrete_path_real[:, 1],
            z=discrete_path_real[:, 2],
            mode='lines',
            line=dict(color='gray', width=3, dash='dot'),
            name='Discrete A* (safe_grid)'
        ))

        fig.add_trace(go.Scatter3d(
            x=smoothed_path_real[:, 0],
            y=smoothed_path_real[:, 1],
            z=smoothed_path_real[:, 2],
            mode='lines',
            line=dict(color='red', width=8),
            name=final_path_label
        ))

        fig.add_trace(go.Scatter3d(
            x=[real_start[0]], y=[real_start[1]], z=[real_start[2]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='diamond'),
            name='Start (projected)'
        ))

        fig.add_trace(go.Scatter3d(
            x=[real_goal[0]], y=[real_goal[1]], z=[real_goal[2]],
            mode='markers',
            marker=dict(color='purple', size=10, symbol='diamond'),
            name='Goal (projected)'
        ))

        roof_clearance_subtitle = (
            "N/A"
            if np.isnan(m_prop["min_roof_clearance_m"])
            else f"{m_prop['min_roof_clearance_m']:.2f}m"
        )

        fig.update_layout(
            title=(f"Risk-Aware Path v2 - Hard Boundary + Verified Smoothing + Simplified Impact Risk<br>"
                   f"<sub>Hard safety: {'PASS' if m_prop['hard_safety_pass'] else 'FAIL'} | "
                   f"Min roof clearance: {roof_clearance_subtitle} | "
                   f"Avg impact risk: {m_prop['avg_impact_risk']:.3f} | "
                   f"Smoothing: {smoothing_state}</sub>"),
            scene=dict(
                xaxis=dict(title='X'), yaxis=dict(title='Y'),
                zaxis=dict(title='Z',
                           range=[mesh.bounds[0][2], mesh.bounds[1][2] + max_extent*0.1]),
                aspectmode='data', bgcolor='white'
            ),
            margin=dict(l=0, r=0, b=0, t=60),
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
            width=1000, height=800,
        )
        fig.update_traces(
            hoverinfo='all',
            hovertemplate="<b>X:</b> %{x:.2f}<br>"
                          "<b>Y:</b> %{y:.2f}<br>"
                          "<b>Z:</b> %{z:.2f}<extra>%{name}</extra>"
        )
        fig.show()
        """,
    )

    _set_cell_source(
        nb,
        26,
        """
USER_START_COORD_CMP = (8, -11, 5)
USER_GOAL_COORD_CMP = (70, -180, 0)


def a_star_standard(safe_grid_hard, start, goal):
    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]

    shape_x, shape_y, shape_z = safe_grid_hard.shape
    gscore = {start: 0}
    came_from = {}
    oheap = [(0, start)]
    close_set = set()

    while oheap:
        curr = heapq.heappop(oheap)[1]
        if curr in close_set:
            continue
        if curr == goal:
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            return path[::-1]
        close_set.add(curr)

        for dx, dy, dz in neighbors:
            nx, ny, nz = curr[0] + dx, curr[1] + dy, curr[2] + dz
            neighbor = (nx, ny, nz)
            if not (0 <= nx < shape_x and 0 <= ny < shape_y and nz >= 0):
                continue
            if nz < shape_z and safe_grid_hard[nx, ny, nz] == 1:
                continue
            if neighbor in close_set:
                continue
            tentative_g = gscore[curr] + math.sqrt(dx * dx + dy * dy + dz * dz)
            if tentative_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = curr
                gscore[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(oheap, (f, neighbor))
    return None


print("1. Prepare projected endpoints for the comparison run...")
s_idx = real_to_voxel_index(USER_START_COORD_CMP)
g_idx = real_to_voxel_index(USER_GOAL_COORD_CMP)
projection_search_radius = max(25, BUFFER_VOXELS * 6)
s_uni, s_proj = find_nearest_empty_with_projection(
    safe_grid, s_idx, max_radius=projection_search_radius
)
g_uni, g_proj = find_nearest_empty_with_projection(
    safe_grid, g_idx, max_radius=projection_search_radius
)
if not np.isfinite(s_proj) or not np.isfinite(g_proj):
    raise RuntimeError("Could not project the comparison start/goal into safe_grid.")

t0 = time.time()
path_std = a_star_standard(safe_grid, s_uni, g_uni)
time_std = time.time() - t0
print(f"   standard A*: {time_std:.2f}s, {len(path_std) if path_std else 'FAIL'} waypoints")

t0 = time.time()
path_prop_raw = a_star_3d_safe_cruise(
    safe_grid,
    dist_map,
    impact_risk_stack,
    s_uni,
    g_uni,
    ALTITUDE_WEIGHT,
    WALL_WEIGHT,
    SAFE_WALL_DISTANCE,
    GROUND_RISK_WEIGHT,
)
time_prop_search = time.time() - t0

path_prop_smooth = None
smooth_report_cmp = None
if path_prop_raw:
    t0 = time.time()
    path_prop_smooth, smooth_report_cmp = smooth_and_verify(
        path_prop_raw, safe_grid, voxel_origin, auto_pitch,
        n_samples=500, max_iterations=3, verbose=False
    )
    time_prop_smooth = time.time() - t0
    time_prop_total = time_prop_search + time_prop_smooth
    print(f"   proposed method (search + smooth): {time_prop_total:.2f}s "
          f"(search {time_prop_search:.2f}s + smooth {time_prop_smooth:.2f}s), "
          f"violations {smooth_report_cmp['final_violations']}/500")
else:
    time_prop_total = time_prop_search
    print("   proposed method failed to find a route")

if path_std:
    p_std_real = voxel_origin + (np.array(path_std) * auto_pitch)
    m_std = get_metrics_final_trajectory(
        p_std_real,
        safe_grid,
        impact_risk_stack,
        roof_height_map_m,
        building_footprint,
        voxel_origin,
        auto_pitch,
    )
else:
    m_std = None

path_prop_raw_real = (
    voxel_origin + (np.array(path_prop_raw) * auto_pitch)
    if path_prop_raw else None
)
m_prop_cmp = (get_metrics_final_trajectory(
    path_prop_smooth,
    safe_grid,
    impact_risk_stack,
    roof_height_map_m,
    building_footprint,
    voxel_origin,
    auto_pitch,
) if path_prop_smooth is not None else None)
        """,
    )

    _set_cell_source(
        nb,
        28,
        """

        print("=== Fair Comparison Result (final trajectory basis) ===")
        print()

        def _format_roof_clearance(metric_dict):
            if metric_dict is None:
                return "FAIL"
            value = metric_dict["min_roof_clearance_m"]
            if np.isnan(value):
                return "N/A"
            return f"{value:.2f}"

        def _format_hard_safety(metric_dict):
            if metric_dict is None:
                return "FAIL"
            return "PASS" if metric_dict["hard_safety_pass"] else "FAIL"

        if m_std and m_prop_cmp:
            print(f"  {'Metric':<30s}  {'Standard A*':>12s}  {'Proposed':>12s}")
            print("  " + "-" * 62)
            print(f"  {'Hard safety pass':<30s}  {_format_hard_safety(m_std):>12s}  {_format_hard_safety(m_prop_cmp):>12s}")
            print(f"  {'Path length (m)':<30s}  {m_std['length_m']:>12.2f}  {m_prop_cmp['length_m']:>12.2f}")
            print(f"  {'Min roof clearance (m)':<30s}  {_format_roof_clearance(m_std):>12s}  {_format_roof_clearance(m_prop_cmp):>12s}")
            print(f"  {'Avg impact risk':<30s}  {m_std['avg_impact_risk']:>12.3f}  {m_prop_cmp['avg_impact_risk']:>12.3f}")
            print(f"  {'Smoothing status':<30s}  {'N/A':>12s}  "
                  f"{('smoothed' if not smooth_report_cmp.get('fallback_used') else 'fallback') if smooth_report_cmp else 'FAIL':>12s}")
            print()
            print(f"  Auxiliary - computation time (s): {time_std:.2f} vs {time_prop_total:.2f}")

        labels = ['Standard A*', 'Proposed']
        colors = ['#636EFA', '#EF553B']

        fig_res = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Path Length (m)',
                'Min Roof Clearance (m)',
                'Avg Impact Risk',
                'Computation Time (s)',
            )
        )

        if m_std and m_prop_cmp:
            roof_plot = [
                0.0 if np.isnan(m_std['min_roof_clearance_m']) else m_std['min_roof_clearance_m'],
                0.0 if np.isnan(m_prop_cmp['min_roof_clearance_m']) else m_prop_cmp['min_roof_clearance_m'],
            ]
            roof_text = [
                'N/A' if np.isnan(m_std['min_roof_clearance_m']) else f"{m_std['min_roof_clearance_m']:.2f}",
                'N/A' if np.isnan(m_prop_cmp['min_roof_clearance_m']) else f"{m_prop_cmp['min_roof_clearance_m']:.2f}",
            ]
            plot_data = [
                ([m_std['length_m'], m_prop_cmp['length_m']], 1, 1, [f"{m_std['length_m']:.2f}", f"{m_prop_cmp['length_m']:.2f}"]),
                (roof_plot, 1, 2, roof_text),
                ([m_std['avg_impact_risk'], m_prop_cmp['avg_impact_risk']], 2, 1, [f"{m_std['avg_impact_risk']:.3f}", f"{m_prop_cmp['avg_impact_risk']:.3f}"]),
                ([time_std, time_prop_total], 2, 2, [f"{time_std:.2f}", f"{time_prop_total:.2f}"]),
            ]
            for values, r, c, text_values in plot_data:
                fig_res.add_trace(
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color=colors,
                        text=text_values,
                        textposition='auto',
                    ),
                    row=r, col=c
                )

        fig_res.update_layout(
            width=1100, height=800,
            title_text='<b>Performance Comparison - Hard Safety and Simplified Risk Metrics</b>',
            showlegend=False, template='plotly_white'
        )
        fig_res.show()


        """,
    )

    _set_cell_source(
        nb,
        30,
        """
        fig_3d = go.Figure(data=[base_building_trace, base_edge_trace])
        fig_3d.update_traces(showlegend=False, selector=dict(name='City Buildings'))
        fig_3d.update_traces(showlegend=False, selector=dict(name='Building Edges'))
        prop_cmp_label = (
            'Proposed (smoothed + verified)'
            if smooth_report_cmp and not smooth_report_cmp.get('fallback_used')
            else 'Proposed (verified fallback)'
        )

        if path_std:
            p_std_real = voxel_origin + (np.array(path_std) * auto_pitch)
            fig_3d.add_trace(go.Scatter3d(
                x=p_std_real[:, 0], y=p_std_real[:, 1], z=p_std_real[:, 2],
                mode='lines', line=dict(color='blue', width=4, dash='dot'),
                name='Standard A*'
            ))

        if path_prop_raw_real is not None:
            fig_3d.add_trace(go.Scatter3d(
                x=path_prop_raw_real[:, 0], y=path_prop_raw_real[:, 1], z=path_prop_raw_real[:, 2],
                mode='lines', line=dict(color='orange', width=3, dash='dash'),
                name='Proposed raw A*'
            ))

        if path_prop_smooth is not None:
            fig_3d.add_trace(go.Scatter3d(
                x=path_prop_smooth[:, 0], y=path_prop_smooth[:, 1], z=path_prop_smooth[:, 2],
                mode='lines', line=dict(color='red', width=8),
                name=prop_cmp_label
            ))

        r_start = voxel_to_real(s_uni)
        r_goal  = voxel_to_real(g_uni)
        fig_3d.add_trace(go.Scatter3d(
            x=[r_start[0]], y=[r_start[1]], z=[r_start[2]],
            mode='markers', marker=dict(color='green', size=10, symbol='diamond'),
            name='Start'
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[r_goal[0]], y=[r_goal[1]], z=[r_goal[2]],
            mode='markers', marker=dict(color='purple', size=10, symbol='diamond'),
            name='Goal'
        ))

        fig_3d.update_layout(
            width=1100, height=800,
            scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'),
                       zaxis=dict(title='Z',
                                  range=[mesh.bounds[0][2], mesh.bounds[1][2]+20]),
                       aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.7)')
        )
        fig_3d.update_traces(
            hoverinfo='all',
            hovertemplate="<b>X:</b> %{x:.2f}<br>"
                          "<b>Y:</b> %{y:.2f}<br>"
                          "<b>Z:</b> %{z:.2f}<extra>%{name}</extra>"
        )
        fig_3d.show()
        """,
    )

    _set_cell_source(
        nb,
        31,
        """
## Next steps
**What this notebook guarantees at the current paper/demo stage**
- The route does **not** pass through `safe_grid`, which already includes the `SAFETY_MARGIN_M` buffer.
- The final smoothed trajectory is re-verified, and a fallback discrete route is used if needed.
- Route evaluation uses **impact-footprint-based ground risk**, not only geometric clearance.
- `ground_risk_map` and `impact_risk_stack` are used as an **offline planning prior** only;
  real-time risk gating is intentionally out of scope for this notebook.

**Known limits (should be stated clearly in the paper)**
- The current ground-risk map is still a **manual, geometry-only demo prior**. It separates buildings from non-building surfaces, but does **not** semantically distinguish roads, sidewalks, plazas, parks, or other open ground classes.
- Therefore, non-building cells should **not** be interpreted as guaranteed safe landing or safe overflight zones; they only share the same simplified baseline prior in this 1st-stage notebook.
- The footprint kernel uses the first-order approximation `r_fp = v_h * sqrt(2h/g) + margin`, i.e. a circular uniform kernel rather than a wind-aware or anisotropic impact distribution.
- Risk slices at high altitude still inherit the assumptions of this simplified footprint model.
- `CRUISE_Z` is a search convenience term, not a physically commanded flight level.

**Recommended next engineering steps**
1. Replace the geometry-only ground-risk prior with OSM or other structured urban land-use data.
2. Add semantic separation inside non-building areas (for example: road / sidewalk / plaza / park / open lot).
3. Standardize the planner output interface so it connects cleanly to Step 1 and Layer 3 demos.
4. Upgrade the impact-footprint model with wind-aware or adaptive kernels for stronger `R_emg` estimates.
        """,
    )

    report_marker = "[REPORT FIGURE EXPORT]"
    _drop_cells_with_marker(nb, report_marker)

    nb["cells"].append(
        _make_markdown_cell(
            f"""
## Report Figure Export {report_marker}
The following cells generate high-resolution report figures for:

- **[그림 4-2]** B-spline 스무딩 전후
- **[그림 4-4]** 본 시스템의 위험 인지형 A* 경로
            """
        )
    )

    nb["cells"].append(
        _make_code_cell(
            f"""
# {report_marker}
REPORT_FIGURE_DIR_NAME = "Diagrams"

def resolve_report_figure_dir():
    candidates = [
        Path("../docs/competition/경로 생성 알고리즘") / REPORT_FIGURE_DIR_NAME,
        Path("../../docs/competition/경로 생성 알고리즘") / REPORT_FIGURE_DIR_NAME,
        Path("docs/competition/경로 생성 알고리즘") / REPORT_FIGURE_DIR_NAME,
        Path(REPORT_FIGURE_DIR_NAME),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate.resolve()
        except Exception:
            continue
    fallback = (Path.cwd() / REPORT_FIGURE_DIR_NAME).resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


REPORT_FIG_DIR = resolve_report_figure_dir()
print(f"Report figures will be saved to: {{REPORT_FIG_DIR}}")


def save_plotly_figure(fig, png_name, html_name=None, scale=2):
    png_path = REPORT_FIG_DIR / png_name
    try:
        fig.write_image(str(png_path), scale=scale)
        print(f"Saved PNG: {{png_path}}")
    except Exception as exc:
        print(f"[warning] PNG export failed: {{exc}}")
    if html_name:
        html_path = REPORT_FIG_DIR / html_name
        fig.write_html(str(html_path))
        print(f"Saved HTML: {{html_path}}")
            """
        )
    )

    nb["cells"].append(
        _make_code_cell(
            f"""
# {report_marker}
# [그림 4-2] B-spline 스무딩 전후
fig_42, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180, constrained_layout=True)

background = building_footprint.T.astype(float)
extent = [
    voxel_origin[0],
    voxel_origin[0] + shape_x * auto_pitch,
    voxel_origin[1],
    voxel_origin[1] + shape_y * auto_pitch,
]

for ax in axes:
    ax.imshow(
        background,
        origin="lower",
        extent=extent,
        cmap="Greys",
        alpha=0.35,
        interpolation="nearest",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

axes[0].plot(
    discrete_path_real[:, 0],
    discrete_path_real[:, 1],
    color="#4C72B0",
    linewidth=2.5,
)
axes[0].scatter(
    discrete_path_real[:, 0],
    discrete_path_real[:, 1],
    color="#4C72B0",
    s=12,
    alpha=0.85,
    label="Discrete waypoints",
)
axes[0].scatter(
    [discrete_path_real[0, 0], discrete_path_real[-1, 0]],
    [discrete_path_real[0, 1], discrete_path_real[-1, 1]],
    c=["green", "purple"],
    s=40,
    zorder=3,
)
axes[0].set_title("Before Smoothing: Discrete A* Path")

axes[1].plot(
    discrete_path_real[:, 0],
    discrete_path_real[:, 1],
    color="gray",
    linewidth=1.2,
    linestyle="--",
    alpha=0.7,
)
axes[1].scatter(
    discrete_path_real[:, 0],
    discrete_path_real[:, 1],
    color="gray",
    s=10,
    alpha=0.55,
)
axes[1].plot(
    smoothed_path_real[:, 0],
    smoothed_path_real[:, 1],
    color="#D62728",
    linewidth=2.8,
)
axes[1].scatter(
    [smoothed_path_real[0, 0], smoothed_path_real[-1, 0]],
    [smoothed_path_real[0, 1], smoothed_path_real[-1, 1]],
    c=["green", "purple"],
    s=40,
    zorder=3,
)
axes[1].set_title(
    "After B-spline Smoothing"
    if not smooth_report.get("fallback_used")
    else "After Verification (Fallback Discrete Path)"
)

fig_42_path = REPORT_FIG_DIR / "Figure_4_2_bspline_before_after.png"
fig_42.savefig(fig_42_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 4-2: {{fig_42_path}}")
plt.show()
            """
        )
    )

    nb["cells"].append(
        _make_code_cell(
            f"""
# {report_marker}
# [그림 4-4] 본 시스템의 위험 인지형 A* 경로
fig_44 = go.Figure(data=[base_building_trace, base_edge_trace])
fig_44.update_traces(showlegend=False, selector=dict(name="City Buildings"))
fig_44.update_traces(showlegend=False, selector=dict(name="Building Edges"))

fig_44.add_trace(go.Scatter3d(
    x=discrete_path_real[:, 0],
    y=discrete_path_real[:, 1],
    z=discrete_path_real[:, 2],
    mode="lines",
    line=dict(color="gray", width=3, dash="dot"),
    name="Discrete A*",
))

fig_44.add_trace(go.Scatter3d(
    x=smoothed_path_real[:, 0],
    y=smoothed_path_real[:, 1],
    z=smoothed_path_real[:, 2],
    mode="lines",
    line=dict(color="red", width=7),
    name=(
        "Risk-aware final path"
        if not smooth_report.get("fallback_used")
        else "Verified fallback path"
    ),
))

real_start = voxel_to_real(start_safe)
real_goal = voxel_to_real(goal_safe)
fig_44.add_trace(go.Scatter3d(
    x=[real_start[0]], y=[real_start[1]], z=[real_start[2]],
    mode="markers",
    marker=dict(color="green", size=8, symbol="diamond"),
    name="Start",
))
fig_44.add_trace(go.Scatter3d(
    x=[real_goal[0]], y=[real_goal[1]], z=[real_goal[2]],
    mode="markers",
    marker=dict(color="purple", size=8, symbol="diamond"),
    name="Goal",
))

fig_44.update_layout(
    width=1100,
    height=800,
    scene=dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(
            title="Z",
            range=[mesh.bounds[0][2], mesh.bounds[1][2] + max_extent * 0.1],
        ),
        aspectmode="data",
        bgcolor="white",
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.75)"),
)

save_plotly_figure(
    fig_44,
    "Figure_4_4_risk_aware_final_path.png",
    html_name="Figure_4_4_risk_aware_final_path.html",
    scale=2,
)
fig_44.show()
            """
        )
    )

    NOTEBOOK_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH.resolve())


if __name__ == "__main__":
    main()
