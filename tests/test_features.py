"""Compare offline features with a reference using the committed C matrices."""
from pathlib import Path
import re
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from features import extract_mfcc


class FeatureTests(unittest.TestCase):
    def test_firmware_matrices_and_log_floor(self):
        header = (ROOT / 'firmware/nexus_kws/mfcc_coeffs.h').read_text()
        def matrix(name, shape):
            body = header.split(f'const float {name}', 1)[1].split('=', 1)[1].split(';', 1)[0]
            return np.array([float(v) for v in re.findall(r'([-+\d.eE]+)f', body)]).reshape(shape)
        mel = matrix('mel_basis', (40, 257))
        dct = matrix('dct_basis', (13, 40))
        rng = np.random.default_rng(42)
        tone = np.sin(2*np.pi*1000*np.arange(16000)/16000).astype(np.float32)
        for audio in [np.zeros(16000, np.float32), tone, rng.normal(0, .1, 16000).astype(np.float32)]:
            windows = np.lib.stride_tricks.sliding_window_view(audio, 512)[::320]
            hann = .5 * (1 - np.cos(2*np.pi*np.arange(512)/512))
            power = abs(np.fft.rfft(windows * hann))**2
            expected = dct @ (10*np.log10(np.maximum(mel @ power.T, 1e-10)))
            actual = extract_mfcc(audio)
            self.assertEqual(actual.shape, (13, 49))
            np.testing.assert_allclose(actual, expected, atol=.002, rtol=.0001)

    def test_padding_and_truncation(self):
        np.testing.assert_array_equal(extract_mfcc(np.zeros(100)), extract_mfcc(np.zeros(17000)))


if __name__ == '__main__':
    unittest.main()
