import unittest

from code.alignment import *


class InflectionShapeTest(unittest.TestCase):
    TESTS = [
        {
            "words": ("hello", "hill"),
            "shapes": InflectionShapes(prefix=False, infix=False, alternation=True, suffix=True)
        }
    ]

    def test_all(self):
        for test in InflectionShapeTest.TESTS:
            self.assertEqual(test["shapes"], get_inflection_shapes(*test["words"]))
