"""Fast-mode tests for PII_INDICATORS only, using temporary dependency stand-ins.
No model download or GPU is needed.

Run: python3 -m unittest discover -s tests
"""
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_stubs = {name: types.ModuleType(name) for name in ("torch", "transformers", "tqdm")}
_stubs["torch"].inference_mode = lambda: (lambda fn: fn)
_stubs["transformers"].AutoModelForCausalLM = object
_stubs["transformers"].AutoTokenizer = object
_stubs["tqdm"].tqdm = lambda x, **_: x
with patch.dict(sys.modules, _stubs):
    PII_INDICATORS = runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "safelog_fast.py")
    )["PII_INDICATORS"]


def selected(line):
    return bool(PII_INDICATORS.search(line))


class BluetoothAndNameFields(unittest.TestCase):
    def test_bluetooth_and_name_lines_match_prefilter(self):
        for line in (
            'D Bluetooth: Connected to "Varun\'s Pixel Buds"',
            "I BtGatt.GattService: onConnected device=Varun's Pixel Buds",
            "D ContactsProvider: display_name=Priya Sharma",
            "bt_stack: device_name=Galaxy Buds2 Pro",
        ):
            with self.subTest(line=line):
                self.assertTrue(selected(line))

    def test_existing_indicators_still_match(self):
        for line in (
            'I WifiService: connected SSID="HomeNet"',
            "D Bluetooth: Connected to Varun's Pixel Buds (aa:bb:cc:dd:ee:ff)",
            "D ContactsProvider: account_name=priya@example.test",
            "D WifiP2p: peer aa:bb:cc:dd:ee:ff",
            "I Mail: sync for priya@example.test",
        ):
            with self.subTest(line=line):
                self.assertTrue(selected(line))

    def test_ordinary_lines_are_skipped(self):
        for line in (
            "I ActivityManager: Start proc 4321:com.android.settings/u0a123 for activity",
            "I SurfaceFlinger: Display 0 HWC layers: 12",
            "I chatty: uid=1000(system) Binder:1234_5 expire 3 lines",
        ):
            with self.subTest(line=line):
                self.assertFalse(selected(line))


if __name__ == "__main__":
    unittest.main()
