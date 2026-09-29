"""CPU model/export checks without retraining or modifying bundled artifacts."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import train_kws_v4 as training


class TrainingTests(unittest.TestCase):
    def test_paths_from_another_directory(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                self.assertEqual(len(list(training.REAL_POSITIVE_DIR.glob('*.wav'))), 51)
                self.assertEqual(training.MODEL_H_INO_PATH, ROOT / 'firmware/nexus_kws/model.h')
                self.assertEqual(training.NORM_H_INO_PATH, ROOT / 'firmware/nexus_kws/mfcc_norm.h')
        finally:
            os.chdir(previous)

    def test_bundled_model_invokes(self):
        interpreter = tf.lite.Interpreter(model_path=str(training.MODEL_TFLITE_PATH))
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        self.assertEqual(input_info['shape'].tolist(), [1, 13, 49, 1])
        self.assertEqual(output_info['shape'].tolist(), [1, 2])
        self.assertEqual(input_info['dtype'], np.int8)
        self.assertEqual(output_info['dtype'], np.int8)
        interpreter.set_tensor(input_info['index'], np.zeros((1, 13, 49, 1), np.int8))
        interpreter.invoke()

    def test_export_to_temporary_artifacts(self):
        tf.keras.utils.set_random_seed(42)
        model = training.build_model()
        self.assertEqual(model.count_params(), 1294)
        features = np.random.default_rng(42).normal(size=(4, 13, 49, 1)).astype(np.float32)
        self.assertEqual(tuple(model(features, training=False).shape), (4, 2))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = dict(MODEL_TFLITE_PATH=tmp/'model.tflite', MODEL_H_INO_PATH=tmp/'firmware/model.h',
                         NORM_H_INO_PATH=tmp/'firmware/mfcc_norm.h', NORM_JSON_PATH=tmp/'stats.json')
            with mock.patch.multiple(training, **paths):
                binary = training.export_model(model, features, [0.0]*13, [1.0]*13)
            self.assertEqual(paths['MODEL_TFLITE_PATH'].read_bytes(), binary)
            for path in paths.values():
                self.assertTrue(path.is_file())
            interpreter = tf.lite.Interpreter(model_content=binary)
            interpreter.allocate_tensors()
            self.assertEqual(interpreter.get_input_details()[0]['dtype'], np.int8)


if __name__ == '__main__':
    unittest.main()
