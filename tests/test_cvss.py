"""Sanity checks for the CVSS v3.1 calculator against well-known reference vectors.

Run with:  python -m unittest discover -s tests
"""
import unittest

from src.cvss import base_score, severity


def parse(vec: str) -> dict:
    return dict(part.split(":") for part in vec.split("/"))


class TestCVSS(unittest.TestCase):
    # (vector, expected base score) - values from the FIRST v3.1 calculator.
    CASES = [
        ("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
        ("AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),
        ("AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
        ("AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),
        ("AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),
        ("AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H", 8.1),
        ("AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N", 4.3),
        ("AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N", 3.3),
        ("AV:N/AC:L/PR:L/UI:N/S:C/C:L/I:L/A:N", 6.4),
        ("AV:P/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N", 4.6),
        ("AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),
    ]

    def test_reference_vectors(self):
        for vec, expected in self.CASES:
            with self.subTest(vec=vec):
                self.assertAlmostEqual(base_score(parse(vec)), expected, places=1)

    def test_severity_bands(self):
        self.assertEqual(severity(3.9), "Low")
        self.assertEqual(severity(4.0), "Medium")
        self.assertEqual(severity(7.0), "High")
        self.assertEqual(severity(9.0), "Critical")


if __name__ == "__main__":
    unittest.main()
