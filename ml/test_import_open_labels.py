import unittest

from import_open_labels import parse_anchor


class AnchorTests(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(parse_anchor("22 секунда"), 22)

    def test_minutes_without_spaces(self):
        self.assertEqual(parse_anchor("1минута 8 секунда"), 68)

    def test_no_event(self):
        self.assertIsNone(parse_anchor(""))

    def test_unknown_format_is_not_silently_zero(self):
        with self.assertRaises(ValueError):
            parse_anchor("неизвестно")
