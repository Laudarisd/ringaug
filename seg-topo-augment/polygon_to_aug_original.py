import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


class IndexPreservingPolygonAugmentor:
    """
    Mask-first augmentation for LabelMe polygons.

    Flow:
    1) Rasterize each source polygon to mask.
    2) Apply augmentation to image + masks (+ original keypoints for index projection).
    3) Extract augmented polygons from masks (clipping/splitting naturally handled by masks).
    4) Project each original index to nearest augmented mask boundary point.

    Outputs per sample:
    - LabelMe JSON: for training.
    - Indexed JSON: source indices + projected boundary points for CAP/debug.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]

    # -----------------------------
    # IO helpers
    # -----------------------------
    def _find_image_for_json(self, json_path: Path, image_dir: Path) -> Path:
        for ext in self.supported_extensions:
            candidate = image_dir / f"{json_path.stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for '{json_path.stem}' in '{image_dir}'.")

    def _load_labelme_data(self, json_path: Path, image_dir: Path) -> Tuple[np.ndarray, Dict[str, Any], str]:
        image_path = self._find_image_for_json(json_path, image_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return image, data, json_path.stem

    # -----------------------------
    # Transform
    # -----------------------------
    def _build_transform(self, h: int, w: int, params: Dict[str, Any]) -> A.Compose:
        crop_scale_range = params.get("crop_scale_range", (0.8, 0.9))
        crop_h = max(8, int(h * random.uniform(*crop_scale_range)))
        crop_w = max(8, int(w * random.uniform(*crop_scale_range)))
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

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

        k = random.randint(2, min(4, len(aug_pool)))
        chosen = random.sample(aug_pool, k)

        return A.Compose(
            chosen,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    # -----------------------------
    # Mask / contour helpers
    # -----------------------------
    @staticmethod
    def _clamp_point(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
        xx = min(max(float(x), 0.0), float(width - 1))
        yy = min(max(float(y), 0.0), float(height - 1))
        return xx, yy

    def _polygon_to_mask(self, points: List[List[float]], height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) < 3:
            return mask
        poly = np.round(np.array(points, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        return mask

    def _extract_labelme_polygons_from_mask(
        self,
        mask: np.ndarray,
        label: str,
        simplify_epsilon: float,
        min_component_area: float,
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """
        Extract split polygons from one augmented mask.
        Returns:
          labelme_shapes_for_this_source_shape,
          dense_contours_for_projection
        """
        shapes: List[Dict[str, Any]] = []
        dense_contours: List[np.ndarray] = []

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            return shapes, dense_contours

        hierarchy = hierarchy[0]

        def to_float_points(cnt: np.ndarray) -> np.ndarray:
            return cnt.reshape(-1, 2).astype(np.float64)

        def nearest_pair(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
            best_i, best_j = 0, 0
            best_d2 = float("inf")
            for i in range(len(a)):
                dx = b[:, 0] - a[i, 0]
                dy = b[:, 1] - a[i, 1]
                d2 = dx * dx + dy * dy
                j = int(np.argmin(d2))
                if float(d2[j]) < best_d2:
                    best_d2 = float(d2[j])
                    best_i, best_j = i, j
            return best_i, best_j

        def make_ring_polygon(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
            # Connect outer and inner contour by shortest bridge pair.
            oi, ii = nearest_pair(outer, inner)
            outer_seq = np.vstack([outer[oi:], outer[: oi + 1]])
            inner_seq = np.vstack([inner[ii:], inner[: ii + 1]])
            # Reverse inner direction for non-self-intersecting bridge traversal.
            inner_seq = inner_seq[::-1]
            bridge_back = outer[oi : oi + 1]
            ring = np.vstack([outer_seq, inner_seq, bridge_back])
            return ring

        visited_outer = set()
        for idx, cnt in enumerate(contours):
            # In CCOMP, parent == -1 means an outer contour.
            if hierarchy[idx][3] != -1:
                continue
            if idx in visited_outer:
                continue

            outer_area = float(cv2.contourArea(cnt))
            if outer_area < min_component_area:
                continue

            outer_dense = to_float_points(cnt)
            if len(outer_dense) < 3:
                continue

            # Include outer boundary for projection.
            dense_contours.append(outer_dense)

            # Collect all hole contours (children in hierarchy chain).
            hole_indices: List[int] = []
            child = int(hierarchy[idx][2])
            while child != -1:
                hole_indices.append(child)
                child = int(hierarchy[child][0])  # next sibling hole

            outer_approx = cv2.approxPolyDP(cnt, simplify_epsilon, True).reshape(-1, 2).astype(np.float64)
            if len(outer_approx) < 3:
                continue

            if not hole_indices:
                labelme_points = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in outer_approx]
                shapes.append(
                    {
                        "label": label,
                        "points": labelme_points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )
                visited_outer.add(idx)
                continue

            # Ring-type: preserve holes by building bridged polygon(s) outer+inner.
            ring_points = outer_approx
            for hid in hole_indices:
                hcnt = contours[hid]
                hole_area = float(cv2.contourArea(hcnt))
                if hole_area < min_component_area:
                    continue

                hole_dense = to_float_points(hcnt)
                if len(hole_dense) < 3:
                    continue
                dense_contours.append(hole_dense)  # include inner boundary for projection too

                hole_approx = cv2.approxPolyDP(hcnt, simplify_epsilon, True).reshape(-1, 2).astype(np.float64)
                if len(hole_approx) < 3:
                    continue

                ring_points = make_ring_polygon(ring_points, hole_approx)

            if len(ring_points) >= 3:
                labelme_points = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in ring_points]
                shapes.append(
                    {
                        "label": label,
                        "points": labelme_points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )
            visited_outer.add(idx)

        return shapes, dense_contours

    @staticmethod
    def _nearest_point_on_contours(x: float, y: float, contours: List[np.ndarray]) -> Tuple[List[float], float, int]:
        """
        Project a point to nearest contour vertex among all components.
        Returns: [px, py], distance, component_index
        """
        best_dist = float("inf")
        best_pt = [float(x), float(y)]
        best_comp = -1

        for ci, contour in enumerate(contours):
            if len(contour) == 0:
                continue
            # Vertex-level nearest projection (simple and stable for index tracking).
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

    # -----------------------------
    # Save outputs
    # -----------------------------
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

        img_path = out_img_dir / aug_img_name
        cv2.imwrite(str(img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        labelme_data = {
            "version": original_data.get("version", "5.5.0"),
            "flags": original_data.get("flags", {}),
            "shapes": labelme_shapes,
            "imagePath": f"..\\images\\{aug_img_name}",
            "imageData": None,
            "imageHeight": int(aug_image.shape[0]),
            "imageWidth": int(aug_image.shape[1]),
        }
        with open(out_json_dir / aug_json_name, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2)

        indexed_data = {
            "version": original_data.get("version", "5.5.0"),
            "flags": original_data.get("flags", {}),
            "imagePath": f"..\\images\\{aug_img_name}",
            "imageData": None,
            "imageHeight": int(aug_image.shape[0]),
            "imageWidth": int(aug_image.shape[1]),
            "shapes": labelme_shapes,
            **indexed_payload,
        }
        with open(out_index_json_dir / aug_json_name, "w", encoding="utf-8") as f:
            json.dump(indexed_data, f, indent=2)

    # -----------------------------
    # Main API
    # -----------------------------
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

        json_files = sorted(label_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {label_dir}")

        saved_count = 0
        for json_path in tqdm(json_files, desc="Augmenting", ncols=90, unit="file"):
            try:
                image, data, base_name = self._load_labelme_data(json_path, image_dir)
                h, w = image.shape[:2]
                raw_shapes = [s for s in data.get("shapes", []) if s.get("shape_type") == "polygon" and len(s.get("points", [])) >= 3]
                if not raw_shapes:
                    continue

                # Source index assignment before augmentation (0..n-1 per source polygon).
                source_original_shapes_indexed: List[Dict[str, Any]] = []
                source_masks: List[np.ndarray] = []
                flat_keypoints: List[Tuple[float, float]] = []
                flat_meta: List[Tuple[int, int]] = []  # (source_shape_index, original_index)

                for src_idx, shape in enumerate(raw_shapes):
                    label = str(shape.get("label", ""))
                    points = shape["points"]

                    source_original_shapes_indexed.append(
                        {
                            "source_shape_index": int(src_idx),
                            "label": label,
                            "original_vertex_count": int(len(points)),
                            "original_points_indexed": {
                                str(i): [round(float(pt[0]), 2), round(float(pt[1]), 2)] for i, pt in enumerate(points)
                            },
                        }
                    )

                    source_masks.append(self._polygon_to_mask(points, h, w))

                    for i, pt in enumerate(points):
                        flat_keypoints.append((float(pt[0]), float(pt[1])))
                        flat_meta.append((src_idx, i))

                if num_augmentations == "random":
                    n_aug = random.randint(1, int(augmentation_params.get("random_aug_per_image", 3)))
                else:
                    n_aug = int(num_augmentations)

                for _ in range(n_aug):
                    transform = self._build_transform(h, w, augmentation_params)
                    transformed = transform(image=image, masks=source_masks, keypoints=flat_keypoints)

                    aug_image = transformed["image"]
                    aug_h, aug_w = aug_image.shape[:2]
                    aug_masks = transformed["masks"]
                    aug_keypoints = transformed["keypoints"]

                    # Group transformed original points back to each source shape index.
                    transformed_keypoints_by_shape: Dict[int, Dict[int, Tuple[float, float]]] = {}
                    for (src_idx, orig_idx), (x, y) in zip(flat_meta, aug_keypoints):
                        if src_idx not in transformed_keypoints_by_shape:
                            transformed_keypoints_by_shape[src_idx] = {}
                        cx, cy = self._clamp_point(float(x), float(y), aug_w, aug_h)
                        transformed_keypoints_by_shape[src_idx][orig_idx] = (cx, cy)

                    labelme_shapes: List[Dict[str, Any]] = []
                    projected_indexed_shapes: List[Dict[str, Any]] = []

                    simplify_epsilon = float(augmentation_params.get("contour_simplify_epsilon", 1.5))
                    min_component_area = float(augmentation_params.get("min_component_area", 12.0))
                    min_mask_pixel_area = int(augmentation_params.get("min_mask_pixel_area", 32))

                    # For each source mask, extract split polygons and project original indices.
                    for src_idx, (shape, mask) in enumerate(zip(raw_shapes, aug_masks)):
                        label = str(shape.get("label", ""))
                        mask_u8 = np.array(mask, dtype=np.uint8)
                        if int(np.count_nonzero(mask_u8)) < min_mask_pixel_area:
                            # Too small after augmentation/crop: skip entirely.
                            continue
                        lm_shapes, dense_contours = self._extract_labelme_polygons_from_mask(
                            mask=mask_u8,
                            label=label,
                            simplify_epsilon=simplify_epsilon,
                            min_component_area=min_component_area,
                        )
                        labelme_shapes.extend(lm_shapes)

                        # Build projected index table for this source shape.
                        transformed_kps = transformed_keypoints_by_shape.get(src_idx, {})
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

                        projected_indexed_shapes.append(
                            {
                                "source_shape_index": int(src_idx),
                                "label": label,
                                "original_vertex_count": int(len(shape.get("points", []))),
                                "projected_vertices": projected_vertices,
                                "component_count_after_clipping": int(len(dense_contours)),
                            }
                        )

                    if not labelme_shapes:
                        continue

                    indexed_payload = {
                        "source_original_shapes_indexed": source_original_shapes_indexed,
                        "projected_indexed_shapes": projected_indexed_shapes,
                    }

                    self._save_outputs(
                        aug_image=aug_image,
                        labelme_shapes=labelme_shapes,
                        indexed_payload=indexed_payload,
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
