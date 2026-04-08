import unittest
import numpy as np

from object_detector_app.utils.app_utils import (
    build_count_overlay_lines,
    color_name_to_rgb,
    count_detected_objects,
    standard_colors,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.colors = color_name_to_rgb()
        self.standard_colors = standard_colors()

    def test_all_colors(self):
        """Test that manual defined colors are also in the matplotlib color name space."""
        color_list = set(sorted(list(self.colors.keys())))
        standard_color_list = set(sorted([color.lower() for color in self.standard_colors]))
        color_common = standard_color_list.intersection(color_list)
        self.assertEqual(len(color_common), len(standard_color_list))

    def test_count_detected_objects_filters_low_confidence(self):
        category_index = {
            1: {"name": "person"},
            3: {"name": "car"},
        }
        classes = np.array([1, 1, 3, 3])
        scores = np.array([0.95, 0.72, 0.40, 0.91])

        object_counts = count_detected_objects(classes, scores, category_index, min_score_thresh=0.5)

        self.assertEqual(object_counts, {"person": 2, "car": 1})

    def test_build_count_overlay_lines_formats_total_and_breakdown(self):
        lines = build_count_overlay_lines({"person": 2, "car": 1})

        self.assertEqual(lines, ["Objects detected: 3", "person: 2", "car: 1"])
