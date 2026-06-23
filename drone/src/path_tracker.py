"""Path tracking interface between the planner and Step 1 detector.

This module sits outside the path planner and converts a planned path
(`smoothed_path_real`) plus the drone's current position into tracking signals
that Step 1 can consume in real time.

Contract
--------
- `cross_track_error`:
    Signed horizontal cross-track error in meters.
    This is the value that should be fed into the Step 1 LSTM because the
    ALFA dataset's `cross_track_error` is a signed lateral signal.
- `cross_track_error_abs`:
    Absolute horizontal cross-track error in meters.
- `cross_track_error_filtered`:
    EMA-filtered absolute horizontal cross-track error.
    Recommended for rule-based thresholds such as `cte_spike`.
- `vertical_error`:
    Absolute vertical deviation in meters.
- `cross_track_error_3d`:
    Full 3D point-to-path distance in meters for diagnostics/plots.

Sign convention
---------------
Positive means "left of the path" in the local XY frame of the active path
segment. The sign is computed from the z-component of the 2D cross product:

    sign = sign(seg_xy x error_xy)

This convention is stable as long as the path is traversed in the planned
direction. If a downstream dataset uses the opposite sign convention, the sign
can be flipped in one place without changing the rest of the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TrackingState:
    """Tracking output for a single time step."""

    cross_track_error: float
    cross_track_error_abs: float
    cross_track_error_filtered: float
    vertical_error: float
    cross_track_error_3d: float
    nearest_waypoint_index: int
    segment_start_index: int
    along_track_distance: float
    progress_ratio: float
    projection_point: np.ndarray
    segment_direction: np.ndarray
    is_off_route: bool = False


class PathTracker:
    """Track the drone against a fixed sampled path.

    The path is assumed to be computed once and then reused as a fixed
    reference. There is no replanning inside this module.
    """

    def __init__(
        self,
        smoothed_path_real: np.ndarray,
        off_route_threshold_m: float = 5.0,
        search_window: int = 50,
        ema_alpha: float = 0.3,
    ) -> None:
        path = np.asarray(smoothed_path_real, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(
                f"smoothed_path_real must have shape (N, 3), got {path.shape}"
            )
        if len(path) < 2:
            raise ValueError("Path must contain at least two points.")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must satisfy 0 < ema_alpha <= 1.")
        if search_window < 0:
            raise ValueError("search_window must be >= 0.")

        self.path = path
        self.n_points = len(path)
        self.off_route_threshold_m = float(off_route_threshold_m)
        self.search_window = int(search_window)
        self.ema_alpha = float(ema_alpha)

        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.segment_lengths = segment_lengths
        self.cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.total_length = float(self.cumulative_lengths[-1])
        if self.total_length <= 0.0:
            raise ValueError("Path total length must be positive.")

        self._last_nearest_idx: Optional[int] = None
        self._ema_cte_abs: Optional[float] = None

    def reset(self) -> None:
        """Reset local tracking memory for a new flight."""

        self._last_nearest_idx = None
        self._ema_cte_abs = None

    def update(self, current_pos) -> TrackingState:
        """Compute tracking state for the current position."""

        current = np.asarray(current_pos, dtype=np.float64).reshape(3)
        nearest_idx = self._find_nearest_in_window(current)
        (
            projection_point,
            segment_direction,
            segment_start_index,
            t_clamped,
            segment_length,
        ) = self._project_to_best_segment(current, nearest_idx)

        error_vec = current - projection_point
        horizontal_error_vec = error_vec[:2]
        horizontal_abs = float(np.linalg.norm(horizontal_error_vec))
        vertical_error = float(abs(error_vec[2]))
        error_3d = float(np.linalg.norm(error_vec))

        signed_horizontal = self._signed_horizontal_error(
            horizontal_error_vec, segment_direction[:2]
        )

        if self._ema_cte_abs is None:
            self._ema_cte_abs = horizontal_abs
        else:
            self._ema_cte_abs = (
                self.ema_alpha * horizontal_abs
                + (1.0 - self.ema_alpha) * self._ema_cte_abs
            )
        filtered_abs = float(self._ema_cte_abs)

        along_track = float(
            self.cumulative_lengths[segment_start_index] + t_clamped * segment_length
        )
        progress_ratio = float(
            np.clip(along_track / self.total_length, 0.0, 1.0)
        )

        self._last_nearest_idx = nearest_idx

        return TrackingState(
            cross_track_error=signed_horizontal,
            cross_track_error_abs=horizontal_abs,
            cross_track_error_filtered=filtered_abs,
            vertical_error=vertical_error,
            cross_track_error_3d=error_3d,
            nearest_waypoint_index=nearest_idx,
            segment_start_index=segment_start_index,
            along_track_distance=along_track,
            progress_ratio=progress_ratio,
            projection_point=projection_point,
            segment_direction=segment_direction,
            is_off_route=filtered_abs > self.off_route_threshold_m,
        )

    @staticmethod
    def build_sensor_fields(state: TrackingState) -> Dict[str, float]:
        """Build the sensor fields expected by Step 1 from a TrackingState."""

        return {
            "cross_track_error": state.cross_track_error,
            "cross_track_error_abs": state.cross_track_error_abs,
            "cross_track_error_filtered": state.cross_track_error_filtered,
            "cross_track_error_v": state.vertical_error,
            "cross_track_error_3d": state.cross_track_error_3d,
            "path_progress_ratio": state.progress_ratio,
        }

    def _find_nearest_in_window(self, current: np.ndarray) -> int:
        """Find the nearest waypoint index, using a local search after init."""

        if self._last_nearest_idx is None or self.search_window == 0:
            dists = np.linalg.norm(self.path - current, axis=1)
            return int(np.argmin(dists))

        lo = max(0, self._last_nearest_idx - self.search_window)
        hi = min(self.n_points, self._last_nearest_idx + self.search_window + 1)
        dists = np.linalg.norm(self.path[lo:hi] - current, axis=1)
        return lo + int(np.argmin(dists))

    def _project_to_best_segment(
        self,
        current: np.ndarray,
        nearest_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
        """Project to the best adjacent segment around the nearest waypoint.

        Segment selection is based on horizontal error first because the Step 1
        contract uses a horizontal signed cross-track error. A 3D distance tie
        breaker is used when horizontal errors are equal.
        """

        candidates = []

        if nearest_idx > 0:
            result = self._project_on_segment_xy(
                current, self.path[nearest_idx - 1], self.path[nearest_idx]
            )
            proj, seg_len, direction, t_clamped, h_abs, d3 = result
            candidates.append((h_abs, d3, proj, direction, nearest_idx - 1, t_clamped, seg_len))

        if nearest_idx < self.n_points - 1:
            result = self._project_on_segment_xy(
                current, self.path[nearest_idx], self.path[nearest_idx + 1]
            )
            proj, seg_len, direction, t_clamped, h_abs, d3 = result
            candidates.append((h_abs, d3, proj, direction, nearest_idx, t_clamped, seg_len))

        if not candidates:
            return (
                self.path[nearest_idx].copy(),
                np.zeros(3, dtype=np.float64),
                max(0, nearest_idx - 1),
                0.0,
                0.0,
            )

        _, _, projection_point, direction, seg_start_idx, t_clamped, seg_len = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        return projection_point, direction, seg_start_idx, t_clamped, seg_len

    @staticmethod
    def _project_on_segment_xy(
        current: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray, float, float, float]:
        """Project onto a segment using the segment's XY footprint.

        Returns
        -------
        projection_point, segment_length, segment_direction, t_clamped,
        horizontal_abs_error, distance_3d
        """

        segment = seg_end - seg_start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-12:
            projection = seg_start.copy()
            horizontal_abs = float(np.linalg.norm((current - projection)[:2]))
            distance_3d = float(np.linalg.norm(current - projection))
            return (
                projection,
                0.0,
                np.zeros(3, dtype=np.float64),
                0.0,
                horizontal_abs,
                distance_3d,
            )

        direction = segment / segment_length
        seg_xy = segment[:2]
        seg_xy_len_sq = float(np.dot(seg_xy, seg_xy))

        if seg_xy_len_sq < 1e-12:
            # Near-vertical segment in XY. Fall back to 3D projection and
            # treat the horizontal sign as unresolved.
            t_raw = float(np.dot(current - seg_start, segment)) / float(
                np.dot(segment, segment)
            )
        else:
            t_raw = float(np.dot(current[:2] - seg_start[:2], seg_xy)) / seg_xy_len_sq

        t_clamped = max(0.0, min(1.0, t_raw))
        projection = seg_start + t_clamped * segment
        error_vec = current - projection
        horizontal_abs = float(np.linalg.norm(error_vec[:2]))
        distance_3d = float(np.linalg.norm(error_vec))
        return (
            projection,
            segment_length,
            direction,
            t_clamped,
            horizontal_abs,
            distance_3d,
        )

    @staticmethod
    def _signed_horizontal_error(
        horizontal_error_vec: np.ndarray,
        segment_direction_xy: np.ndarray,
    ) -> float:
        """Compute signed horizontal CTE using the local path frame."""

        horizontal_abs = float(np.linalg.norm(horizontal_error_vec))
        if horizontal_abs < 1e-12:
            return 0.0

        seg_norm = float(np.linalg.norm(segment_direction_xy))
        if seg_norm < 1e-12:
            return horizontal_abs

        seg_xy_unit = segment_direction_xy / seg_norm
        cross_z = float(
            seg_xy_unit[0] * horizontal_error_vec[1]
            - seg_xy_unit[1] * horizontal_error_vec[0]
        )
        if abs(cross_z) < 1e-12:
            return 0.0
        sign = 1.0 if cross_z > 0.0 else -1.0
        return sign * horizontal_abs


def _self_test() -> None:
    """Lightweight smoke tests for the path tracker."""

    print("=" * 68)
    print("PathTracker self-test")
    print("=" * 68)

    n = 101
    path = np.zeros((n, 3), dtype=np.float64)
    path[:, 0] = np.linspace(0.0, 100.0, n)
    path[:, 2] = 10.0

    tracker = PathTracker(path, off_route_threshold_m=5.0, search_window=20)
    print(f"path points: {n}, total length: {tracker.total_length:.2f} m")

    state = tracker.update([50.0, 0.0, 10.0])
    print("\n[Test 1] on-route point")
    print(f"  signed cte: {state.cross_track_error:.6f}")
    print(f"  vertical  : {state.vertical_error:.6f}")
    assert abs(state.cross_track_error) < 1e-9
    assert abs(state.vertical_error) < 1e-9

    state = tracker.update([50.0, 3.0, 10.0])
    print("\n[Test 2] left-of-path lateral deviation")
    print(f"  signed cte: {state.cross_track_error:.6f} (expect +3)")
    assert abs(state.cross_track_error - 3.0) < 1e-6

    state = tracker.update([50.0, -3.0, 10.0])
    print("\n[Test 3] right-of-path lateral deviation")
    print(f"  signed cte: {state.cross_track_error:.6f} (expect -3)")
    assert abs(state.cross_track_error + 3.0) < 1e-6

    state = tracker.update([50.0, 0.0, 17.0])
    print("\n[Test 4] vertical-only deviation")
    print(f"  signed cte: {state.cross_track_error:.6f} (expect 0)")
    print(f"  vertical  : {state.vertical_error:.6f} (expect 7)")
    assert abs(state.cross_track_error) < 1e-6
    assert abs(state.vertical_error - 7.0) < 1e-6

    tracker_mid = PathTracker(path, off_route_threshold_m=5.0, search_window=20)
    state = tracker_mid.update([50.5, 2.0, 10.0])
    print("\n[Test 5] midpoint projection and along-track")
    print(f"  signed cte : {state.cross_track_error:.6f} (expect +2)")
    print(f"  along-track: {state.along_track_distance:.6f} (expect 50.5)")
    assert abs(state.cross_track_error - 2.0) < 1e-6
    assert abs(state.along_track_distance - 50.5) < 1e-6

    tracker_ema = PathTracker(
        path, off_route_threshold_m=5.0, search_window=20, ema_alpha=0.3
    )
    for _ in range(3):
        tracker_ema.update([20.0, 0.0, 10.0])
    state = tracker_ema.update([20.0, 8.0, 10.0])
    print("\n[Test 6] one-step spike absorbed by EMA")
    print(f"  abs cte     : {state.cross_track_error_abs:.6f}")
    print(f"  filtered cte: {state.cross_track_error_filtered:.6f}")
    print(f"  off-route   : {state.is_off_route}")
    assert state.cross_track_error_abs == 8.0
    assert abs(state.cross_track_error_filtered - 2.4) < 1e-6
    assert state.is_off_route is False

    tracker_seq = PathTracker(path, off_route_threshold_m=5.0, search_window=10)
    prev_idx = -1
    for x in range(0, 90, 5):
        state = tracker_seq.update([float(x), 0.5, 10.0])
        assert state.nearest_waypoint_index >= prev_idx
        prev_idx = state.nearest_waypoint_index
    print("\n[Test 7] monotonic local search")
    print(f"  final nearest index: {prev_idx}")

    helix_t = np.linspace(0.0, 4.0 * np.pi, 200)
    helix = np.column_stack(
        [20.0 * np.cos(helix_t), 20.0 * np.sin(helix_t), 2.0 * helix_t]
    )
    tracker_helix = PathTracker(helix, off_route_threshold_m=3.0, search_window=30)
    idx = 50
    state = tracker_helix.update(helix[idx] + np.array([0.0, 0.5, 0.2]))
    print("\n[Test 8] curved 3D path")
    print(f"  abs cte     : {state.cross_track_error_abs:.6f}")
    print(f"  vertical    : {state.vertical_error:.6f}")
    assert state.cross_track_error_abs < 1.0
    assert state.vertical_error < 1.0

    print("\nAll tests passed.")


if __name__ == "__main__":
    _self_test()
