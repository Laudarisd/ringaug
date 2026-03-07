import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


class IndexPreservingPolygonAugmentor:
    """Mask-first polygon augmentation with index projection + safe index-order repair."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]

    # Find the matching image file for a LabelMe JSON.
    def _find_image_for_json(self, json_path: Path, image_dir: Path) -> Path:
        for ext in self.supported_extensions:
            candidate = image_dir / f"{json_path.stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for '{json_path.stem}' in '{image_dir}'.")

    # Load RGB image + LabelMe data.
    def _load_labelme_data(self, json_path: Path, image_dir: Path) -> Tuple[np.ndarray, Dict[str, Any], str]:
        image_path = self._find_image_for_json(json_path, image_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), data, json_path.stem

    # Build a random transform pipeline for one augmentation sample.
    def _build_transform(self, h: int, w: int, params: Dict[str, Any]) -> A.Compose:
        crop_scale_range = params.get("crop_scale_range", (0.8, 0.9))
        crop_h = min(max(8, int(h * random.uniform(*crop_scale_range))), h)
        crop_w = min(max(8, int(w * random.uniform(*crop_scale_range))), w)

        aug_pool = [
            A.Rotate(limit=params.get("angle_limit", (-20, 20)), p=params.get("p_rotate", 0.8)),
            A.HorizontalFlip(p=params.get("p_flip_h", 0.5)),
            A.VerticalFlip(p=params.get("p_flip_v", 0.5)),
            A.Affine(
                scale=params.get("scale_limit", (0.8, 1.2)),
                translate_percent=params.get("translate_limit", (-0.1, 0.1)),
                p=params.get("p_affine", 0.8),
            ),
            A.RandomCrop(height=crop_h, width=crop_w, p=params.get("p_crop", 0.7)),
            A.RandomBrightnessContrast(
                brightness_limit=params.get("brightness_limit", (-0.1, 0.1)),
                contrast_limit=params.get("contrast_limit", (-0.1, 0.1)),
                p=params.get("p_brightness", 0.4),
            ),
        ]
        return A.Compose(
            random.sample(aug_pool, random.randint(2, min(4, len(aug_pool)))),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    # Resolve fixed or random augment count.
    def _resolve_aug_count(self, num_augmentations: Union[int, str], augmentation_params: Dict[str, Any]) -> int:
        if num_augmentations == "random":
            return random.randint(1, int(augmentation_params.get("random_aug_per_image", 3)))
        return int(num_augmentations)

    # Build a LabelMe shape payload.
    @staticmethod
    def _make_labelme_shape(label: str, points: List[List[float]]) -> Dict[str, Any]:
        return {
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        }

    # Round contour points to LabelMe-friendly float pairs.
    @staticmethod
    def _to_labelme_points(points: np.ndarray) -> List[List[float]]:
        return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in points]

    # Keep valid polygon shapes and flatten source keypoints for index tracking.
    def _prepare_source_shapes(
        self, data: Dict[str, Any], h: int, w: int, overlap_eps: float
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[np.ndarray],
        List[Tuple[float, float]],
        List[Tuple[int, int]],
        Dict[int, Dict[str, Any]],
    ]:
        raw_shapes = [
            s for s in data.get("shapes", []) if s.get("shape_type") == "polygon" and len(s.get("points", [])) >= 3
        ]

        source_original_shapes_indexed: List[Dict[str, Any]] = []
        source_masks: List[np.ndarray] = []
        flat_keypoints: List[Tuple[float, float]] = []
        flat_meta: List[Tuple[int, int]] = []
        source_overlap_constraints_by_shape: Dict[int, Dict[str, Any]] = {}

        for src_idx, shape in enumerate(raw_shapes):
            label = str(shape.get("label", ""))
            points = shape["points"]
            overlap_groups, overlap_pairs = self._detect_overlapped_vertices(points, eps=overlap_eps)
            overlap_pair_records = self._build_overlap_pair_records(points, overlap_pairs)

            source_original_shapes_indexed.append(
                {
                    "source_shape_index": int(src_idx),
                    "label": label,
                    "original_vertex_count": int(len(points)),
                    "original_points_indexed": {
                        str(i): [round(float(pt[0]), 2), round(float(pt[1]), 2)] for i, pt in enumerate(points)
                    },
                    "overlap_groups": [list(g) for g in overlap_groups],
                    "overlap_pairs": [list(p) for p in overlap_pairs],
                    "overlap_pair_records": overlap_pair_records,
                }
            )
            source_masks.append(self._polygon_to_mask(points, h, w))
            source_overlap_constraints_by_shape[src_idx] = {
                "overlap_groups": overlap_groups,
                "overlap_pairs": overlap_pairs,
                "overlap_pair_records": overlap_pair_records,
            }

            for i, pt in enumerate(points):
                flat_keypoints.append((float(pt[0]), float(pt[1])))
                flat_meta.append((src_idx, i))

        return (
            raw_shapes,
            source_original_shapes_indexed,
            source_masks,
            flat_keypoints,
            flat_meta,
            source_overlap_constraints_by_shape,
        )

    # Regroup transformed flat keypoints by shape and original vertex index.
    @staticmethod
    def _group_keypoints_by_shape(
        flat_meta: List[Tuple[int, int]], aug_keypoints: List[Tuple[float, float]]
    ) -> Dict[int, Dict[int, Tuple[float, float]]]:
        grouped: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for (src_idx, orig_idx), (x, y) in zip(flat_meta, aug_keypoints):
            grouped.setdefault(src_idx, {})[orig_idx] = (float(x), float(y))
        return grouped

    # Read all repair thresholds from augmentation params.
    @staticmethod
    def _read_repair_params(augmentation_params: Dict[str, Any]) -> Dict[str, float]:
        overlap_eps = float(
            augmentation_params.get("source_overlap_eps", augmentation_params.get("repair_dedupe_eps", 0.5))
        )
        return {
            "simplify_epsilon": float(augmentation_params.get("contour_simplify_epsilon", 1.5)),
            "min_component_area": float(augmentation_params.get("min_component_area", 12.0)),
            "min_mask_pixel_area": int(augmentation_params.get("min_mask_pixel_area", 32)),
            "min_repair_area": float(augmentation_params.get("min_repair_polygon_area", 1.0)),
            "dedupe_eps": float(augmentation_params.get("repair_dedupe_eps", 0.5)),
            "max_projection_distance_for_repair": float(
                augmentation_params.get("max_projection_distance_for_repair", 4.0)
            ),
            "min_retained_vertex_ratio_for_repair": float(
                augmentation_params.get("min_retained_vertex_ratio_for_repair", 0.7)
            ),
            "source_overlap_eps": overlap_eps,
        }

    # Find near-overlapped source vertices and keep pair/group constraints.
    @staticmethod
    def _detect_overlapped_vertices(points: List[List[float]], eps: float) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        n = len(points)
        if n < 2:
            return [], []

        # Build all close candidates, then keep disjoint best pairs only.
        candidates: List[Tuple[float, int, int]] = []
        for i in range(n):
            xi, yi = float(points[i][0]), float(points[i][1])
            for j in range(i + 1, n):
                xj, yj = float(points[j][0]), float(points[j][1])
                if abs(xi - xj) <= eps and abs(yi - yj) <= eps:
                    dx = xi - xj
                    dy = yi - yj
                    candidates.append((dx * dx + dy * dy, i, j))

        if not candidates:
            return [], []

        candidates.sort(key=lambda t: t[0])
        used = set()
        overlap_pairs: List[Tuple[int, int]] = []
        for _, i, j in candidates:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            overlap_pairs.append((i, j))

        overlap_groups = [[i, j] for i, j in overlap_pairs]
        overlap_groups.sort(key=lambda g: min(g))
        overlap_pairs.sort()
        return overlap_groups, overlap_pairs

    # Build per-pair bridge records so each overlap pair is projected separately.
    @staticmethod
    def _build_overlap_pair_records(
        points: List[List[float]], overlap_pairs: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        n = len(points)
        records: List[Dict[str, Any]] = []
        for pair_id, (i, j) in enumerate(sorted(overlap_pairs)):
            xi, yi = float(points[i][0]), float(points[i][1])
            xj, yj = float(points[j][0]), float(points[j][1])
            records.append(
                {
                    "pair_id": int(pair_id),
                    "indices": [int(i), int(j)],
                    "source_anchor": [round((xi + xj) / 2.0, 2), round((yi + yj) / 2.0, 2)],
                    "connections": {
                        str(i): [int((i - 1) % n), int((i + 1) % n)],
                        str(j): [int((j - 1) % n), int((j + 1) % n)],
                    },
                }
            )
        return records

    # Force overlapped index groups to share one projected point.
    def _enforce_overlap_projection(
        self,
        projected_vertices: List[Dict[str, Any]],
        overlap_groups: List[List[int]],
        overlap_pair_records: List[Dict[str, Any]],
        dense_contours: List[np.ndarray],
        min_pair_separation: float,
    ) -> None:
        if not projected_vertices or not overlap_groups:
            return

        by_index = {int(pv["original_index"]): pv for pv in projected_vertices}
        used_overlap_points: List[List[float]] = []

        pair_sources = overlap_pair_records or [{"indices": list(group)} for group in overlap_groups]
        for pair in pair_sources:
            indices = [int(v) for v in pair.get("indices", [])]
            members = [by_index[idx] for idx in indices if idx in by_index]
            if len(members) < 2:
                continue

            # Use one shared transformed overlap anchor for the whole group.
            tx = float(np.mean([float(pv["transformed_point"][0]) for pv in members]))
            ty = float(np.mean([float(pv["transformed_point"][1]) for pv in members]))
            if dense_contours:
                shared_point, _, shared_comp = self._nearest_point_on_contours_avoid_points(
                    tx,
                    ty,
                    dense_contours,
                    avoid_points=used_overlap_points,
                    min_separation=min_pair_separation,
                )
            else:
                shared_point, shared_comp = [round(tx, 2), round(ty, 2)], -1
            used_overlap_points.append(shared_point)

            for pv in members:
                pv["transformed_point"] = [round(tx, 2), round(ty, 2)]
                pv["projected_point"] = shared_point
                pv["projected_component_index"] = shared_comp
                dx = float(pv["transformed_point"][0]) - float(shared_point[0])
                dy = float(pv["transformed_point"][1]) - float(shared_point[1])
                pv["projection_distance"] = round(float(np.sqrt(dx * dx + dy * dy)), 6)

    # Reproject non-pair vertices if they collide with remembered overlap positions.
    def _evict_nonpair_vertices_from_overlap_points(
        self,
        projected_vertices: List[Dict[str, Any]],
        overlap_pairs: List[Tuple[int, int]],
        dense_contours: List[np.ndarray],
        eps: float,
    ) -> None:
        if not projected_vertices or not overlap_pairs:
            return

        protected_indices = set()
        for a, b in overlap_pairs:
            protected_indices.add(int(a))
            protected_indices.add(int(b))

        by_index = {int(pv["original_index"]): pv for pv in projected_vertices}
        protected_points: List[List[float]] = []
        for idx in sorted(protected_indices):
            if idx in by_index:
                protected_points.append([float(by_index[idx]["projected_point"][0]), float(by_index[idx]["projected_point"][1])])

        for pv in projected_vertices:
            idx = int(pv["original_index"])
            if idx in protected_indices:
                continue

            px, py = float(pv["projected_point"][0]), float(pv["projected_point"][1])
            collided = False
            for ax, ay in protected_points:
                if abs(px - ax) <= eps and abs(py - ay) <= eps:
                    collided = True
                    break
            if not collided:
                continue

            tx = float(pv["transformed_point"][0])
            ty = float(pv["transformed_point"][1])
            if dense_contours:
                new_pt, dist, comp = self._nearest_point_on_contours_avoid_points(
                    tx,
                    ty,
                    dense_contours,
                    avoid_points=protected_points,
                    min_separation=eps,
                )
                pv["projected_point"] = new_pt
                pv["projected_component_index"] = int(comp)
                pv["projection_distance"] = round(float(dist), 6) if np.isfinite(dist) else None

    # Project transformed source vertices to nearest contour points.
    def _project_vertices(
        self,
        transformed_kps: Dict[int, Tuple[float, float]],
        dense_contours: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        projected_vertices: List[Dict[str, Any]] = []
        for orig_idx in sorted(transformed_kps.keys()):
            tx, ty = transformed_kps[orig_idx]
            if dense_contours:
                proj_pt, dist, comp_idx = self._nearest_point_on_contours(tx, ty, dense_contours)
            else:
                proj_pt, dist, comp_idx = [round(tx, 2), round(ty, 2)], float("inf"), -1

            projected_vertices.append(
                {
                    "original_index": int(orig_idx),
                    "transformed_point": [round(tx, 2), round(ty, 2)],
                    "projected_point": proj_pt,
                    "projection_distance": round(float(dist), 6) if np.isfinite(dist) else None,
                    "projected_component_index": int(comp_idx),
                }
            )
        return projected_vertices

    # Safely repair polygon order using projected original index order.
    def _apply_index_order_repair(
        self,
        lm_shapes: List[Dict[str, Any]],
        dense_contours: List[np.ndarray],
        projected_vertices: List[Dict[str, Any]],
        overlap_groups: List[List[int]],
        overlap_pairs: List[Tuple[int, int]],
        overlap_pair_records: List[Dict[str, Any]],
        min_repair_area: float,
        dedupe_eps: float,
        source_overlap_eps: float,
        max_projection_distance_for_repair: float,
        min_retained_vertex_ratio_for_repair: float,
        aug_w: int,
        aug_h: int,
    ) -> Tuple[bool, bool, Any, Any, Any]:
        do_repair = len(lm_shapes) == 1 and len(dense_contours) == 1 and len(projected_vertices) >= 3
        if not do_repair:
            return False, False, "broken_or_multi_component", None, None

        # Keep originally overlapped vertices overlapped after projection.
        self._enforce_overlap_projection(
            projected_vertices,
            overlap_groups,
            overlap_pair_records,
            dense_contours,
            min_pair_separation=source_overlap_eps,
        )
        self._evict_nonpair_vertices_from_overlap_points(
            projected_vertices=projected_vertices,
            overlap_pairs=overlap_pairs,
            dense_contours=dense_contours,
            eps=dedupe_eps,
        )

        ordered = sorted(projected_vertices, key=lambda d: d["original_index"])
        repaired_pairs = []
        for pv in ordered:
            tx, ty = float(pv["transformed_point"][0]), float(pv["transformed_point"][1])
            cx, cy = self._clamp_point(tx, ty, aug_w, aug_h)
            repaired_pairs.append((int(pv["original_index"]), [round(cx, 2), round(cy, 2)]))
        # Keep full original index connectivity when overlap/bridge vertices are present.
        if not overlap_groups:
            repaired_pairs = self._dedupe_consecutive_points_with_constraints(
                repaired_pairs,
                eps=dedupe_eps,
                preserve_pairs=overlap_pairs,
            )
        repaired_indices = [idx for idx, _ in repaired_pairs]
        repaired_pts = [pt for _, pt in repaired_pairs]

        projection_distances = [
            float(pv["projection_distance"]) for pv in projected_vertices if pv.get("projection_distance") is not None
        ]
        max_proj_dist = max(projection_distances) if projection_distances else float("inf")
        retained_ratio = len(repaired_pts) / float(len(projected_vertices)) if projected_vertices else 0.0

        repair_max_projection_distance = round(float(max_proj_dist), 6)
        repair_retained_vertex_ratio = round(float(retained_ratio), 6)

        has_dup = self._has_unexpected_near_duplicate_points(
            repaired_indices,
            repaired_pts,
            overlap_groups=overlap_groups,
            eps=dedupe_eps,
        )
        is_valid = self._is_valid_polygon_points(repaired_pts, min_area=min_repair_area)
        is_simple = self._is_simple_polygon(repaired_pts)
        projection_ok = max_proj_dist <= max_projection_distance_for_repair
        retention_ok = retained_ratio >= min_retained_vertex_ratio_for_repair

        if not has_dup and is_valid and is_simple and projection_ok and retention_ok:
            lm_shapes[0]["points"] = repaired_pts
            return True, True, None, repair_max_projection_distance, repair_retained_vertex_ratio

        if has_dup:
            reason = "duplicate_projected_vertices"
        elif not is_valid:
            reason = "invalid_repaired_polygon"
        elif not is_simple:
            reason = "self_intersection_detected"
        elif not projection_ok:
            reason = "projection_distance_too_large"
        else:
            reason = "too_many_vertices_collapsed"

        return True, False, reason, repair_max_projection_distance, repair_retained_vertex_ratio

    # Clamp a point to image bounds.
    @staticmethod
    def _clamp_point(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
        return min(max(float(x), 0.0), float(width - 1)), min(max(float(y), 0.0), float(height - 1))

    # Remove consecutive near-duplicate points.
    @staticmethod
    def _dedupe_consecutive_points(pts: List[List[float]], eps: float = 0.5) -> List[List[float]]:
        if not pts:
            return pts
        out = [pts[0]]
        for p in pts[1:]:
            if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
                out.append(p)
        if len(out) >= 2 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
            out.pop()
        return out

    # Remove consecutive duplicates except index pairs that must stay overlapped.
    @staticmethod
    def _dedupe_consecutive_points_with_constraints(
        indexed_pts: List[Tuple[int, List[float]]],
        eps: float,
        preserve_pairs: List[Tuple[int, int]],
    ) -> List[Tuple[int, List[float]]]:
        if not indexed_pts:
            return indexed_pts

        keep_pair = {(min(a, b), max(a, b)) for a, b in preserve_pairs}
        out = [indexed_pts[0]]
        for idx, pt in indexed_pts[1:]:
            prev_idx, prev_pt = out[-1]
            pair_key = (min(prev_idx, idx), max(prev_idx, idx))
            is_near_dup = abs(pt[0] - prev_pt[0]) <= eps and abs(pt[1] - prev_pt[1]) <= eps
            if is_near_dup and pair_key not in keep_pair:
                continue
            out.append((idx, pt))

        if len(out) >= 2:
            first_idx, first_pt = out[0]
            last_idx, last_pt = out[-1]
            pair_key = (min(first_idx, last_idx), max(first_idx, last_idx))
            is_near_dup = abs(first_pt[0] - last_pt[0]) <= eps and abs(first_pt[1] - last_pt[1]) <= eps
            if is_near_dup and pair_key not in keep_pair:
                out.pop()
        return out

    # Check polygon has at least 3 points and area threshold.
    @staticmethod
    def _is_valid_polygon_points(pts: List[List[float]], min_area: float = 1.0) -> bool:
        if pts is None or len(pts) < 3:
            return False
        area = float(cv2.contourArea(np.array(pts, dtype=np.float32).reshape(-1, 1, 2)))
        return area >= float(min_area)

    # Check if any two points are near-identical.
    @staticmethod
    def _has_near_duplicate_points(pts: List[List[float]], eps: float = 0.5) -> bool:
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pts[i][0] - pts[j][0]) <= eps and abs(pts[i][1] - pts[j][1]) <= eps:
                    return True
        return False

    # Check duplicate points except those belonging to expected overlap groups.
    @staticmethod
    def _has_unexpected_near_duplicate_points(
        indices: List[int],
        pts: List[List[float]],
        overlap_groups: List[List[int]],
        eps: float = 0.5,
    ) -> bool:
        allowed_pairs = set()
        for group in overlap_groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    allowed_pairs.add((min(a, b), max(a, b)))

        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pts[i][0] - pts[j][0]) > eps or abs(pts[i][1] - pts[j][1]) > eps:
                    continue
                key = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                if key not in allowed_pairs:
                    return True
        return False

    # Check if two line segments intersect.
    @staticmethod
    def _segments_intersect(
        a: List[float], b: List[float], c: List[float], d: List[float], eps: float = 1e-9
    ) -> bool:
        def orient(p: List[float], q: List[float], r: List[float]) -> int:
            v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(v) <= eps:
                return 0
            return 1 if v > 0 else 2

        def on_segment(p: List[float], q: List[float], r: List[float]) -> bool:
            return (
                min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
                and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
            )

        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(a, c, b):
            return True
        if o2 == 0 and on_segment(a, d, b):
            return True
        if o3 == 0 and on_segment(c, a, d):
            return True
        if o4 == 0 and on_segment(c, b, d):
            return True
        return False

    # Check self-intersection (adjacent edges can share endpoints).
    @classmethod
    def _is_simple_polygon(cls, pts: List[List[float]]) -> bool:
        n = len(pts)
        if n < 3:
            return False
        for i in range(n):
            a1, a2 = pts[i], pts[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or j == (i + 1) % n or (j + 1) % n == i:
                    continue
                b1, b2 = pts[j], pts[(j + 1) % n]
                if cls._segments_intersect(a1, a2, b1, b2):
                    return False
        return True

    # Rasterize polygon points to binary mask.
    def _polygon_to_mask(self, points: List[List[float]], height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) < 3:
            return mask
        cv2.fillPoly(mask, [np.round(np.array(points, dtype=np.float32)).astype(np.int32)], 255)
        return mask

    # Extract LabelMe polygons and dense contours from one augmented mask.
    def _extract_labelme_polygons_from_mask(
        self,
        mask: np.ndarray,
        label: str,
        simplify_epsilon: float,
        min_component_area: float,
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        shapes: List[Dict[str, Any]] = []
        dense_contours: List[np.ndarray] = []

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            return shapes, dense_contours
        hierarchy = hierarchy[0]

        def to_float_points(cnt: np.ndarray) -> np.ndarray:
            return cnt.reshape(-1, 2).astype(np.float64)

        def nearest_pair(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
            best_i, best_j, best_d2 = 0, 0, float("inf")
            for i in range(len(a)):
                dx = b[:, 0] - a[i, 0]
                dy = b[:, 1] - a[i, 1]
                d2 = dx * dx + dy * dy
                j = int(np.argmin(d2))
                if float(d2[j]) < best_d2:
                    best_i, best_j, best_d2 = i, j, float(d2[j])
            return best_i, best_j

        def make_ring_polygon(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
            n_out, n_in = len(outer), len(inner)
            if n_out < 3 or n_in < 3:
                return outer
            outer_a, inner_b = nearest_pair(outer, inner)
            outer_seq = np.vstack([outer[outer_a:], outer[: outer_a + 1]])
            inner_seq = np.vstack([inner[inner_b:], inner[: inner_b + 1]])[::-1]
            outer_a_pt = outer[outer_a : outer_a + 1]
            inner_b_pt = inner[inner_b : inner_b + 1]
            return np.vstack([outer_seq, inner_b_pt, inner_seq, outer_a_pt])

        for idx, cnt in enumerate(contours):
            if hierarchy[idx][3] != -1:
                continue
            if float(cv2.contourArea(cnt)) < min_component_area:
                continue

            outer_dense = to_float_points(cnt)
            if len(outer_dense) < 3:
                continue
            dense_contours.append(outer_dense)

            hole_indices: List[int] = []
            child = int(hierarchy[idx][2])
            while child != -1:
                hole_indices.append(child)
                child = int(hierarchy[child][0])

            outer_approx = cv2.approxPolyDP(cnt, simplify_epsilon, True).reshape(-1, 2).astype(np.float64)
            if len(outer_approx) < 3:
                continue

            if not hole_indices:
                shapes.append(self._make_labelme_shape(label, self._to_labelme_points(outer_approx)))
                continue

            ring_points = outer_approx
            for hid in hole_indices:
                hcnt = contours[hid]
                if float(cv2.contourArea(hcnt)) < min_component_area:
                    continue

                hole_dense = to_float_points(hcnt)
                if len(hole_dense) < 3:
                    continue
                dense_contours.append(hole_dense)

                hole_approx = cv2.approxPolyDP(hcnt, simplify_epsilon, True).reshape(-1, 2).astype(np.float64)
                if len(hole_approx) < 3:
                    continue
                ring_points = make_ring_polygon(ring_points, hole_approx)

            if len(ring_points) >= 3:
                shapes.append(self._make_labelme_shape(label, self._to_labelme_points(ring_points)))

        return shapes, dense_contours

    # Project one point to nearest contour vertex.
    @staticmethod
    def _nearest_point_on_contours(x: float, y: float, contours: List[np.ndarray]) -> Tuple[List[float], float, int]:
        best_dist = float("inf")
        best_pt = [float(x), float(y)]
        best_comp = -1

        for ci, contour in enumerate(contours):
            if len(contour) == 0:
                continue
            dx = contour[:, 0] - float(x)
            dy = contour[:, 1] - float(y)
            d2 = dx * dx + dy * dy
            idx = int(np.argmin(d2))
            dist = float(np.sqrt(d2[idx]))
            if dist < best_dist:
                best_dist = dist
                best_pt = [round(float(contour[idx, 0]), 2), round(float(contour[idx, 1]), 2)]
                best_comp = ci

        return best_pt, best_dist, best_comp

    # Project to nearest contour point while keeping distance from already-used points.
    @staticmethod
    def _nearest_point_on_contours_avoid_points(
        x: float,
        y: float,
        contours: List[np.ndarray],
        avoid_points: List[List[float]],
        min_separation: float,
    ) -> Tuple[List[float], float, int]:
        best_dist = float("inf")
        best_pt = [float(x), float(y)]
        best_comp = -1
        min_sep2 = float(min_separation) * float(min_separation)

        for ci, contour in enumerate(contours):
            if len(contour) == 0:
                continue
            for pt in contour:
                px = float(pt[0])
                py = float(pt[1])
                if avoid_points:
                    too_close = False
                    for ax, ay in avoid_points:
                        dx0 = px - float(ax)
                        dy0 = py - float(ay)
                        if dx0 * dx0 + dy0 * dy0 < min_sep2:
                            too_close = True
                            break
                    if too_close:
                        continue

                dx = px - float(x)
                dy = py - float(y)
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist < best_dist:
                    best_dist = dist
                    best_pt = [round(px, 2), round(py, 2)]
                    best_comp = ci

        if best_comp == -1:
            return IndexPreservingPolygonAugmentor._nearest_point_on_contours(x, y, contours)
        return best_pt, best_dist, best_comp

    # Split mask contours into outer and inner rings.
    @staticmethod
    def _split_outer_inner_contours(mask: np.ndarray, min_component_area: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            return [], []
        hierarchy = hierarchy[0]

        def to_float(cnt: np.ndarray) -> np.ndarray:
            return cnt.reshape(-1, 2).astype(np.float64)

        outer: List[np.ndarray] = []
        inner: List[np.ndarray] = []
        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            if area < min_component_area:
                continue
            if hierarchy[idx][3] == -1:
                outer.append(to_float(cnt))
            else:
                inner.append(to_float(cnt))
        return outer, inner

    # Draw polyline contours with a specific color.
    @staticmethod
    def _draw_contours_rgb(canvas: np.ndarray, contours: List[np.ndarray], color: Tuple[int, int, int], thickness: int) -> None:
        for cnt in contours:
            if len(cnt) < 2:
                continue
            poly = np.round(cnt).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=thickness)

    # Draw indexed points for visibility.
    @staticmethod
    def _draw_indexed_points_rgb(
        canvas: np.ndarray,
        points_by_index: Dict[int, List[float]],
        point_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
    ) -> None:
        for idx in sorted(points_by_index.keys()):
            x, y = points_by_index[idx]
            cx, cy = int(round(float(x))), int(round(float(y)))
            cv2.circle(canvas, (cx, cy), 3, point_color, -1)
            cv2.putText(canvas, str(idx), (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    # Draw polygon connections using index order.
    @staticmethod
    def _draw_index_connections_rgb(
        canvas: np.ndarray,
        points_by_index: Dict[int, List[float]],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        keys = sorted(points_by_index.keys())
        if len(keys) < 2:
            return
        for i in range(len(keys)):
            a = keys[i]
            b = keys[(i + 1) % len(keys)]
            p1 = points_by_index[a]
            p2 = points_by_index[b]
            cv2.line(
                canvas,
                (int(round(float(p1[0]))), int(round(float(p1[1])))),
                (int(round(float(p2[0]))), int(round(float(p2[1])))),
                color,
                thickness,
            )

    # Save side-by-side index debug image for non-broken cases.
    def _save_bridge_debug_plot(
        self,
        base_name: str,
        aug_iter: int,
        src_idx: int,
        class_label: str,
        original_image: np.ndarray,
        aug_image: np.ndarray,
        source_points: List[List[float]],
        final_shape_points: List[List[float]],
        source_mask: np.ndarray,
        aug_mask: np.ndarray,
        projected_vertices: List[Dict[str, Any]],
        overlap_pairs: List[Tuple[int, int]],
        min_component_area: float,
        out_debug_dir: Path,
    ) -> None:
        out_debug_dir.mkdir(parents=True, exist_ok=True)
        left = np.array(original_image, copy=True)
        right = np.array(aug_image, copy=True)

        left_idx_pts = {i: [float(pt[0]), float(pt[1])] for i, pt in enumerate(source_points)}
        if final_shape_points and len(final_shape_points) >= 3:
            right_idx_pts = {i: [float(pt[0]), float(pt[1])] for i, pt in enumerate(final_shape_points)}
        else:
            right_idx_pts = {
                int(pv["original_index"]): [float(pv["projected_point"][0]), float(pv["projected_point"][1])]
                for pv in projected_vertices
            }

        self._draw_index_connections_rgb(left, left_idx_pts, color=(0, 255, 255), thickness=2)
        self._draw_index_connections_rgb(right, right_idx_pts, color=(0, 255, 255), thickness=2)
        self._draw_indexed_points_rgb(left, left_idx_pts, point_color=(255, 0, 0), text_color=(255, 0, 0))
        self._draw_indexed_points_rgb(right, right_idx_pts, point_color=(255, 0, 0), text_color=(255, 0, 0))

        h = max(left.shape[0], right.shape[0])
        w_left = left.shape[1]
        w_right = right.shape[1]
        panel = np.zeros((h + 30, w_left + w_right + 20, 3), dtype=np.uint8)
        panel[: left.shape[0], :w_left] = left
        panel[: right.shape[0], w_left + 20 : w_left + 20 + w_right] = right
        cv2.putText(
            panel, "Previous Index (left)", (8, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            panel,
            "Augmented Index (right)",
            (w_left + 28, h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        safe_label = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(class_label))
        out_name = f"{base_name}_aug{aug_iter:02d}_shape{src_idx:02d}_{safe_label}_debug.png"
        cv2.imwrite(str(out_debug_dir / out_name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    # Save augmented image, standard LabelMe JSON, and indexed debug JSON.
    def _save_outputs(
        self,
        aug_image: np.ndarray,
        labelme_shapes: List[Dict[str, Any]],
        indexed_payload: Dict[str, Any],
        original_data: Dict[str, Any],
        base_name: str,
        out_img_dir: Path,
        out_json_dir: Path,
        out_index_json_dir: Path,
    ) -> None:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_json_dir.mkdir(parents=True, exist_ok=True)
        out_index_json_dir.mkdir(parents=True, exist_ok=True)

        suffix = uuid.uuid4().hex[:6]
        aug_img_name = f"{base_name}_{suffix}_aug.png"
        aug_json_name = f"{base_name}_{suffix}_aug.json"
        aug_img_path = out_img_dir / aug_img_name

        cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        # Keep LabelMe imagePath relative to each JSON location.
        json_rel_image_path = Path(os.path.relpath(aug_img_path, start=out_json_dir)).as_posix()
        index_rel_image_path = Path(os.path.relpath(aug_img_path, start=out_index_json_dir)).as_posix()

        base_json_common = {
            "version": original_data.get("version", "5.5.0"),
            "flags": original_data.get("flags", {}),
            "imageData": None,
            "imageHeight": int(aug_image.shape[0]),
            "imageWidth": int(aug_image.shape[1]),
        }

        with open(out_json_dir / aug_json_name, "w", encoding="utf-8") as f:
            json.dump({**base_json_common, "imagePath": json_rel_image_path, "shapes": labelme_shapes}, f, indent=2)

        with open(out_index_json_dir / aug_json_name, "w", encoding="utf-8") as f:
            json.dump(
                {**base_json_common, "imagePath": index_rel_image_path, "shapes": labelme_shapes, **indexed_payload},
                f,
                indent=2,
            )

    # Build one index/debug record for a source polygon.
    @staticmethod
    def _build_projected_shape_payload(
        src_idx: int,
        label: str,
        shape: Dict[str, Any],
        projected_vertices: List[Dict[str, Any]],
        dense_contours: List[np.ndarray],
        repair_attempted: bool,
        repair_applied: bool,
        repair_skip_reason: Any,
        repair_max_projection_distance: Any,
        repair_retained_vertex_ratio: Any,
    ) -> Dict[str, Any]:
        return {
            "source_shape_index": int(src_idx),
            "label": label,
            "original_vertex_count": int(len(shape.get("points", []))),
            "projected_vertices": projected_vertices,
            "component_count_after_clipping": int(len(dense_contours)),
            "repair_attempted": repair_attempted,
            "repair_applied": repair_applied,
            "repair_skip_reason": repair_skip_reason,
            "repair_max_projection_distance": repair_max_projection_distance,
            "repair_retained_vertex_ratio": repair_retained_vertex_ratio,
        }

    # End-to-end dataset augmentation API.
    def augment_dataset(
        self,
        data_dir: Union[str, Path],
        json_dir: Union[str, Path],
        save_img_dir: Union[str, Path],
        save_json_dir: Union[str, Path],
        save_index_json_dir: Union[str, Path],
        num_augmentations: Union[int, str],
        augmentation_params: Dict[str, Any],
    ) -> None:
        image_dir = Path(data_dir)
        label_dir = Path(json_dir)
        out_img_dir = Path(save_img_dir)
        out_json_dir = Path(save_json_dir)
        out_index_json_dir = Path(save_index_json_dir)
        out_debug_dir = out_index_json_dir.parent / "debug_bridge_plots"
        out_debug_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(label_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {label_dir}")

        saved_count = 0
        debug_plot_saved_count = 0
        debug_bridge_candidate_count = 0
        debug_nonbroken_bridge_count = 0
        for json_path in tqdm(json_files, desc="Augmenting", ncols=90, unit="file"):
            try:
                image, data, base_name = self._load_labelme_data(json_path, image_dir)
                h, w = image.shape[:2]
                repair_params = self._read_repair_params(augmentation_params)

                (
                    raw_shapes,
                    source_original_shapes_indexed,
                    source_masks,
                    flat_keypoints,
                    flat_meta,
                    source_overlap_constraints_by_shape,
                ) = self._prepare_source_shapes(data, h, w, overlap_eps=repair_params["source_overlap_eps"])
                if not raw_shapes:
                    continue

                n_aug = self._resolve_aug_count(num_augmentations, augmentation_params)

                for aug_iter in range(n_aug):
                    transformed = self._build_transform(h, w, augmentation_params)(
                        image=image, masks=source_masks, keypoints=flat_keypoints
                    )

                    aug_image = transformed["image"]
                    aug_masks = transformed["masks"]
                    transformed_keypoints_by_shape = self._group_keypoints_by_shape(flat_meta, transformed["keypoints"])

                    labelme_shapes: List[Dict[str, Any]] = []
                    projected_indexed_shapes: List[Dict[str, Any]] = []

                    for src_idx, (shape, mask) in enumerate(zip(raw_shapes, aug_masks)):
                        label = str(shape.get("label", ""))
                        mask_u8 = np.array(mask, dtype=np.uint8)
                        if int(np.count_nonzero(mask_u8)) < repair_params["min_mask_pixel_area"]:
                            continue

                        lm_shapes, dense_contours = self._extract_labelme_polygons_from_mask(
                            mask=mask_u8,
                            label=label,
                            simplify_epsilon=repair_params["simplify_epsilon"],
                            min_component_area=repair_params["min_component_area"],
                        )

                        projected_vertices = self._project_vertices(
                            transformed_keypoints_by_shape.get(src_idx, {}),
                            dense_contours,
                        )
                        overlap_constraints = source_overlap_constraints_by_shape.get(
                            src_idx,
                            {"overlap_groups": [], "overlap_pairs": [], "overlap_pair_records": []},
                        )

                        (
                            repair_attempted,
                            repair_applied,
                            repair_skip_reason,
                            repair_max_projection_distance,
                            repair_retained_vertex_ratio,
                        ) = self._apply_index_order_repair(
                            lm_shapes=lm_shapes,
                            dense_contours=dense_contours,
                            projected_vertices=projected_vertices,
                            overlap_groups=overlap_constraints["overlap_groups"],
                            overlap_pairs=overlap_constraints["overlap_pairs"],
                            overlap_pair_records=overlap_constraints["overlap_pair_records"],
                            min_repair_area=repair_params["min_repair_area"],
                            dedupe_eps=repair_params["dedupe_eps"],
                            source_overlap_eps=repair_params["source_overlap_eps"],
                            max_projection_distance_for_repair=repair_params["max_projection_distance_for_repair"],
                            min_retained_vertex_ratio_for_repair=repair_params[
                                "min_retained_vertex_ratio_for_repair"
                            ],
                            aug_w=aug_image.shape[1],
                            aug_h=aug_image.shape[0],
                        )

                        labelme_shapes.extend(lm_shapes)
                        projected_indexed_shapes.append(
                            self._build_projected_shape_payload(
                                src_idx=src_idx,
                                label=label,
                                shape=shape,
                                projected_vertices=projected_vertices,
                                dense_contours=dense_contours,
                                repair_attempted=repair_attempted,
                                repair_applied=repair_applied,
                                repair_skip_reason=repair_skip_reason,
                                repair_max_projection_distance=repair_max_projection_distance,
                                repair_retained_vertex_ratio=repair_retained_vertex_ratio,
                            )
                        )

                        # Save index-only debug plot for each processed shape.
                        if len(lm_shapes) == 1:
                            debug_nonbroken_bridge_count += 1
                        self._save_bridge_debug_plot(
                            base_name=base_name,
                            aug_iter=aug_iter,
                            src_idx=src_idx,
                            class_label=label,
                            original_image=image,
                            aug_image=aug_image,
                            source_points=shape.get("points", []),
                            final_shape_points=(lm_shapes[0].get("points", []) if lm_shapes else []),
                            source_mask=np.array(source_masks[src_idx], dtype=np.uint8),
                            aug_mask=mask_u8,
                            projected_vertices=projected_vertices,
                            overlap_pairs=overlap_constraints["overlap_pairs"],
                            min_component_area=repair_params["min_component_area"],
                            out_debug_dir=out_debug_dir,
                        )
                        debug_plot_saved_count += 1

                    if not labelme_shapes:
                        continue

                    self._save_outputs(
                        aug_image=aug_image,
                        labelme_shapes=labelme_shapes,
                        indexed_payload={
                            "source_original_shapes_indexed": source_original_shapes_indexed,
                            "projected_indexed_shapes": projected_indexed_shapes,
                        },
                        original_data=data,
                        base_name=base_name,
                        out_img_dir=out_img_dir,
                        out_json_dir=out_json_dir,
                        out_index_json_dir=out_index_json_dir,
                    )
                    saved_count += 1

            except Exception as e:
                if self.debug:
                    tqdm.write(f"[ERROR] {json_path.name}: {e}")

        print(f"[DONE] Saved {saved_count} augmented samples.")
        if self.debug:
            print(
                "[DEBUG] Bridge plot summary: "
                f"bridge_candidates={debug_bridge_candidate_count}, "
                f"nonbroken_bridge={debug_nonbroken_bridge_count}, "
                f"saved={debug_plot_saved_count}, "
                f"dir='{out_debug_dir}'"
            )


if __name__ == "__main__":
    params = {
        "crop_scale_range": (0.8, 0.9),
        "angle_limit": (-30, 30),
        "p_rotate": 0.9,
        "p_flip_h": 0.2,
        "p_flip_v": 0.1,
        "p_affine": 0.8,
        "scale_limit": (0.7, 1.3),
        "translate_limit": (-0.1, 0.1),
        "p_crop": 0.7,
        "brightness_limit": (-0.1, 0.1),
        "contrast_limit": (-0.1, 0.1),
        "p_brightness": 0.4,
        "random_aug_per_image": 3,
        "contour_simplify_epsilon": 1.5,
        "min_component_area": 12.0,
        "min_mask_pixel_area": 32,
        "min_repair_polygon_area": 1.0,
        "repair_dedupe_eps": 0.5,
        "source_overlap_eps": 0.5,
        "max_projection_distance_for_repair": 4.0,
        "min_retained_vertex_ratio_for_repair": 0.7,
    }

    augmentor = IndexPreservingPolygonAugmentor(debug=False)
    augmentor.augment_dataset(
        data_dir="seg-topo-augment/images",
        json_dir="seg-topo-augment/json",
        save_img_dir="seg-topo-augment/augmented/images",
        save_json_dir="seg-topo-augment/augmented/json",
        save_index_json_dir="seg-topo-augment/augmented/augmented_index_json",
        num_augmentations=2,
        augmentation_params=params,
    )
