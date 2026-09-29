"""Offline integrity and downloader regression checks; no third-party packages."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download', ROOT / 'scripts/download_speech_commands.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class ArtifactTests(unittest.TestCase):
    def test_model_header_matches_binary(self):
        header = (ROOT / 'firmware/nexus_kws/model.h').read_text()
        binary = (ROOT / 'training/model.tflite').read_bytes()
        self.assertEqual(bytes(int(v, 16) for v in re.findall(r'0x([0-9a-fA-F]{2})', header)), binary)
        self.assertIn(f'model_tflite_len = {len(binary)};', header)
        self.assertEqual(binary[4:8], b'TFL3')

    def test_normalization_header_matches_json(self):
        stats = json.loads((ROOT / 'training/mfcc_norm_stats.json').read_text())
        header = (ROOT / 'firmware/nexus_kws/mfcc_norm.h').read_text()
        for name in ('mean', 'std'):
            values = re.search(r'mfcc_' + name + r'\[13\] = \{([^}]+)', header)[1]
            values = [float(v.strip().rstrip('f')) for v in values.split(',')]
            self.assertEqual(len(stats[name]), 13)
            for actual, expected in zip(values, stats[name]):
                self.assertAlmostEqual(actual, expected, delta=0.000051)
                if name == 'std':
                    self.assertGreater(actual, 0)

    def test_manifest_and_recording_format(self):
        manifest = json.loads((ROOT / 'docs/artifacts.json').read_text())
        for path, checksum in manifest['sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), checksum)
        for category, count in [('real_positive', 51), ('real_negative', 60)]:
            files = list((ROOT / 'dataset' / category).glob('*.wav'))
            self.assertEqual(len(files), count)
            for path in files:
                with wave.open(str(path)) as wav:
                    self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 16000, 24000))


class DownloadTests(unittest.TestCase):
    def test_checksum_failure_prevents_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'bad.tar.gz'
            archive.write_bytes(b'incomplete')
            with mock.patch.multiple(download, DATASET_DIR=str(Path(tmp)/'data'), ARCHIVE_PATH=str(archive)), mock.patch.object(download, 'extract_archive') as extract:
                with self.assertRaises(SystemExit):
                    download.main()
                extract.assert_not_called()
                self.assertTrue(archive.exists())

    def test_unsafe_archive_rejected(self):
        for name, symlink in [('../outside', False), ('link', True)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'test.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    member = tarfile.TarInfo(name)
                    if symlink:
                        member.type = tarfile.SYMTYPE
                        member.linkname = '../outside'
                    tar.addfile(member, io.BytesIO())
                with self.assertRaises(tarfile.FilterError):
                    download.extract_archive(archive, Path(tmp)/'data')
                self.assertFalse((Path(tmp)/'outside').exists())

    def test_valid_archive_extracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'test.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                member = tarfile.TarInfo('yes/sample.wav')
                member.size = 3
                tar.addfile(member, io.BytesIO(b'abc'))
            download.extract_archive(archive, Path(tmp)/'data')
            self.assertEqual((Path(tmp)/'data/yes/sample.wav').read_bytes(), b'abc')


if __name__ == '__main__':
    unittest.main()
