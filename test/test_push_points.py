"""Contour, 2F-85 footprint and five-candidate selection regressions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import cv2
import numpy as np

from action_primitives import PUSH_POINT_CLEARANCE_M, TARGET_EDGE_OFFSET_M, PushPrimitiveSet
from environment import ACTION_HEIGHT_MODE, EnvironmentConfig
from features import HEIGHTMAP_RESOLUTION_M, Observation, extract_features
from gripper import Gripper2F85
from push_points import (CANDIDATE_SHIFT_M, PushPointError, extract_target_contour,
                         gripper_footprint_mask, sample_target_edges, select_push_point)
from scene import WORKSPACE_LIMITS, WORKSPACE_REFERENCE_Z_M


ORIGIN = WORKSPACE_LIMITS[:2, 0]
RESOLUTION = HEIGHTMAP_RESOLUTION_M
FOOTPRINT = dict(origin_xy_m=ORIGIN, resolution_m=RESOLUTION,
                 along_push_m=Gripper2F85.CLOSED_TIP_ALONG_PUSH_M,
                 across_push_m=Gripper2F85.CLOSED_TIP_ACROSS_PUSH_M)


def observation_from_mask(mask):
    geometry = extract_target_contour(mask)
    heightmap = np.where(mask, .03, 0).astype(np.float32)
    center = None if geometry is None else ORIGIN+(geometry.center_pixel_xy+.5)*RESOLUTION
    return Observation(heightmap, mask, extract_features(heightmap, mask), center,
                       None if geometry is None else .03,
                       None if geometry is None else geometry.points)


class PushPointTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((250, 250), dtype=np.uint8)
        self.mask[115:136, 115:136] = 1
        self.observation = observation_from_mask(self.mask)
        self.planner = PushPrimitiveSet(WORKSPACE_REFERENCE_Z_M)

    def test_offset_is_half_narrow_width_plus_5mm(self):
        self.assertAlmostEqual(PUSH_POINT_CLEARANCE_M, .005)
        self.assertAlmostEqual(TARGET_EDGE_OFFSET_M, .013)
        self.assertAlmostEqual(TARGET_EDGE_OFFSET_M,
                               Gripper2F85.CLOSED_TIP_ALONG_PUSH_M/2+.005)

    def test_edges_are_contour_intersections_not_bounding_box(self):
        mask = np.zeros_like(self.mask)
        polygon = np.rint(cv2.boxPoints(((125., 125.), (40., 18.), 30.))).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        contour = extract_target_contour(mask).points
        edges = sample_target_edges(contour)
        x, y, w, h = cv2.boundingRect(contour)
        self.assertEqual(edges.shape, (4, 2))
        # A rotated rectangle's central ray exits before its bbox extremes.
        self.assertLess(edges[0, 0], x+w-3)
        self.assertLess(edges[1, 1], y+h-3)
        self.assertGreater(edges[2, 0], x+2)
        self.assertGreater(edges[3, 1], y+2)
        outward = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)))
        for hit in edges-.5*outward:
            self.assertEqual(cv2.pointPolygonTest(contour, tuple(float(v) for v in hit), False), 0)

    def test_edge_distance_is_measured_from_cell_outer_face(self):
        edges = sample_target_edges(self.observation.target_contour_px)
        # Rectangle mask occupies pixels 115..135 inclusive. Its outer
        # faces are half a pixel beyond those cell centres, symmetrically.
        np.testing.assert_allclose(edges[:, 0], (135.5, 125, 114.5, 125))
        np.testing.assert_allclose(edges[:, 1], (125, 135.5, 125, 114.5))
        for action in range(4):
            primitive = self.planner.plan_observation(action, self.observation)
            outward = np.array((np.cos(action*np.pi/2), np.sin(action*np.pi/2)))
            edge_world = ORIGIN+(edges[action]+.5)*RESOLUTION
            nominal = primitive.point_selection.candidates_xy_m[0]
            gap = (nominal-edge_world)@outward-Gripper2F85.CLOSED_TIP_ALONG_PUSH_M/2
            self.assertAlmostEqual(gap, .005)

    def test_reference_contour_preprocessing_and_centroid(self):
        geometry = extract_target_contour(self.mask)
        blurred = cv2.GaussianBlur(self.mask*255, (3, 3), 0)
        threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = max(contours, key=lambda c: (len(c), cv2.contourArea(c)))
        np.testing.assert_array_equal(geometry.points, expected)
        moments = cv2.moments(expected)
        np.testing.assert_array_equal(geometry.center_pixel_xy,
                                      (int(moments['m10']/moments['m00']),
                                       int(moments['m01']/moments['m00'])))
        self.assertIsNone(extract_target_contour(np.zeros_like(self.mask)))
        noise = np.zeros_like(self.mask)
        noise[10, 10] = 1
        self.assertIsNone(extract_target_contour(noise))

    def test_all_five_candidates_are_scored_and_selectable(self):
        center = ORIGIN+(np.array((125., 125.))+.5)*RESOLUTION
        shifts = CANDIDATE_SHIFT_M*np.array(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)))
        for angle in np.arange(4)*np.pi/2:
            for wanted in range(5):
                with self.subTest(angle=angle, wanted=wanted):
                    heightmap = np.full((250, 250), .04)
                    mask = gripper_footprint_mask(heightmap.shape, center+shifts[wanted], angle, **FOOTPRINT)
                    heightmap[mask] = 0
                    selection = select_push_point(heightmap, center, angle, **FOOTPRINT)
                    np.testing.assert_allclose(selection.candidates_xy_m, center+shifts)
                    self.assertEqual(selection.selected_index, wanted)
                    self.assertEqual(selection.max_heights_m[wanted], 0)
                    np.testing.assert_allclose(np.delete(selection.max_heights_m, wanted), .04)

    def test_score_uses_maximum_not_mean_and_ties_prefer_original(self):
        center = self.observation.target_center_xy
        heights = np.zeros((250, 250))
        first = select_push_point(heights, center, 0., **FOOTPRINT)
        self.assertEqual(first.selected_index, 0)
        # One tall cell dominates any footprint that includes it.
        footprint = gripper_footprint_mask(heights.shape, center, 0., **FOOTPRINT)
        y, x = np.argwhere(footprint)[0]
        heights[y, x] = .04
        selection = select_push_point(heights, center, 0., **FOOTPRINT)
        self.assertAlmostEqual(selection.max_heights_m[0], .04)
        self.assertNotEqual(selection.selected_index, 0)
        self.assertEqual(selection.max_heights_m[selection.selected_index], 0)

    def test_rectangular_footprint_rotates_with_direction(self):
        center = self.observation.target_center_xy
        mask_x = gripper_footprint_mask((250, 250), center, 0., **FOOTPRINT)
        mask_y = gripper_footprint_mask((250, 250), center, np.pi/2, **FOOTPRINT)
        ys, xs = np.nonzero(mask_x)
        self.assertGreater(np.ptp(ys), np.ptp(xs))
        ys, xs = np.nonzero(mask_y)
        self.assertGreater(np.ptp(xs), np.ptp(ys))
        np.testing.assert_array_equal(mask_x,
                                      gripper_footprint_mask((250, 250), center, np.pi, **FOOTPRINT))

    def test_only_low_actions_sample_and_rank_points(self):
        # High actions work with neither a contour nor a usable heightmap.
        self.observation.target_contour_px = None
        self.observation.heightmap = None
        with patch('action_primitives.sample_target_edges', side_effect=AssertionError('No edge sampling')):
            with patch('action_primitives.select_push_point', side_effect=AssertionError('No point selection')):
                for action in range(4, 8):
                    primitive = self.planner.plan_observation(action, self.observation)
                    np.testing.assert_array_equal(primitive.start_position_m[:2], self.observation.target_center_xy)
                    self.assertIsNone(primitive.point_selection)
                    self.assertAlmostEqual(primitive.start_position_m[2], WORKSPACE_REFERENCE_Z_M+.035)
        with self.assertRaises(PushPointError):
            self.planner.plan_observation(0, self.observation)

    def test_low_action_nominal_offset_height_and_push_direction(self):
        pixels = sample_target_edges(self.observation.target_contour_px)
        boundaries = ORIGIN+(pixels+.5)*RESOLUTION
        for action in range(4):
            primitive = self.planner.plan_observation(action, self.observation)
            outward = np.array((np.cos(action*np.pi/2), np.sin(action*np.pi/2)))
            selection = primitive.point_selection
            np.testing.assert_allclose(selection.candidates_xy_m[0], boundaries[action]+.013*outward)
            self.assertEqual(selection.selected_index, 0)
            np.testing.assert_allclose(primitive.end_position_m-primitive.start_position_m,
                                       np.r_[-.08*outward, 0], atol=1e-12)
            self.assertAlmostEqual(primitive.start_position_m[2], WORKSPACE_REFERENCE_Z_M+.025)
            # The inward-shifted candidate overlaps the visible target; its
            # local max is target height, unlike the empty original point.
            inward_index = (3, 4, 1, 2)[action]
            self.assertAlmostEqual(selection.max_heights_m[inward_index], .03, places=6)
            self.assertEqual(selection.max_heights_m[0], 0)

    def test_outside_and_nonfinite_footprints_are_not_zero_scored(self):
        heights = np.zeros((250, 250))
        selection = select_push_point(heights, ORIGIN+(.005, .25), 0., **FOOTPRINT)
        self.assertTrue(np.isinf(selection.max_heights_m[0]))
        self.assertEqual(selection.selected_index, 1)
        with self.assertRaises(PushPointError):
            select_push_point(heights, ORIGIN-1., 0., **FOOTPRINT)
        with self.assertRaises(PushPointError):
            select_push_point(np.full_like(heights, np.nan), self.observation.target_center_xy, 0., **FOOTPRINT)

    def test_old_bbox_point_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different push-point convention'):
            EnvironmentConfig.from_checkpoint({'action_height_mode': ACTION_HEIGHT_MODE})
        config = EnvironmentConfig()
        self.assertEqual(EnvironmentConfig.from_checkpoint(vars(config)), config)


if __name__ == '__main__':
    unittest.main()
