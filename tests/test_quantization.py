import unittest

from pyramidkv.quantization import (
    KIVI_AXIS_KEY,
    KIVI_AXIS_VALUE,
    build_quantized_cache_config,
    normalize_quant_method,
)


class QuantizationConfigTest(unittest.TestCase):
    def test_normalize_empty_quant_method(self):
        self.assertIsNone(normalize_quant_method(None))
        self.assertIsNone(normalize_quant_method(""))
        self.assertIsNone(normalize_quant_method("none"))

    def test_rejects_unknown_quant_method(self):
        with self.assertRaises(ValueError):
            normalize_quant_method("unknown")

    def test_builds_kivi_asymmetric_config(self):
        config = build_quantized_cache_config("kivi", nbits=2, residual_length=128)
        self.assertEqual(config["backend"], "hqq")
        self.assertEqual(config["axis_key"], KIVI_AXIS_KEY)
        self.assertEqual(config["axis_value"], KIVI_AXIS_VALUE)
        self.assertEqual(config["nbits"], 2)
        self.assertEqual(config["residual_length"], 128)
        self.assertEqual(config["q_group_size"], 64)

    def test_allows_axis_override_for_experiments(self):
        config = build_quantized_cache_config(
            "kvquant",
            nbits=4,
            residual_length=256,
            axis_key=0,
            axis_value=1,
            q_group_size=32,
        )
        self.assertEqual(config["axis_key"], 0)
        self.assertEqual(config["axis_value"], 1)
        self.assertEqual(config["q_group_size"], 32)


if __name__ == "__main__":
    unittest.main()
