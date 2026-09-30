"""Protocol regression tests without a downloaded Vosk model or ESP32."""
import contextlib
import importlib.util
import io
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import wave


class StreamingServerTests(unittest.TestCase):
    def run_stream(self, chunks, results, final_text):
        recognizer = Mock()
        received = []

        def accept_pcm(pcm):
            self.assertEqual(len(pcm) % 2, 0)
            received.append(pcm)
            return True

        recognizer.AcceptWaveform.side_effect = accept_pcm
        recognizer.Result.side_effect = results
        recognizer.FinalResult.return_value = '{"text": "' + final_text + '"}'
        vosk = types.ModuleType('vosk')
        vosk.Model = Mock()
        vosk.KaldiRecognizer = Mock(return_value=recognizer)
        spec = importlib.util.spec_from_file_location(
            'asr_server_under_test', Path(__file__).with_name('asr_server.py'))
        server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, vosk=vosk):
            spec.loader.exec_module(server)

        client = Mock()
        client.recv.side_effect = chunks
        listener = Mock()
        listener.accept.side_effect = [(client, ('127.0.0.1', 12345)), KeyboardInterrupt]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            server.__file__ = str(Path(directory) / 'asr_server.py')
            with patch.object(server.socket, 'socket', return_value=listener), contextlib.redirect_stdout(output):
                server.run_server()
            recordings = list((Path(directory) / 'recordings').glob('command_*.wav'))
            self.assertEqual(len(recordings), 1)
            with wave.open(str(recordings[0]), 'rb') as recording:
                self.assertEqual((recording.getnchannels(), recording.getsampwidth(),
                                  recording.getframerate()), (1, 2, 16000))
                pcm = recording.readframes(recording.getnframes())
        vosk.Model.assert_called_once_with(lang='en-in')
        recognizer.FinalResult.assert_called_once()
        client.close.assert_called_once()
        listener.close.assert_called_once()
        self.assertEqual(pcm, b''.join(received))
        return pcm, output.getvalue()

    def test_fragmented_samples_and_accumulated_results(self):
        pcm, output = self.run_stream(
            [b'\x01', b'\x02\x03\x04\x05', b'\x06', b''],
            ['{"text": "nexus turn"}', '{"text": "on the"}'], 'lights')
        self.assertEqual(pcm, b'\x01\x02\x03\x04\x05\x06')
        self.assertIn('Spoken Command: "turn on the lights"', output)

    def test_timeout_finalizes_and_drops_incomplete_sample(self):
        pcm, output = self.run_stream(
            [b'\x01\x02\x03', socket.timeout()], ['{"text": ""}'], 'successful command')
        self.assertEqual(pcm, b'\x01\x02')
        self.assertIn('incomplete trailing PCM sample', output)
        self.assertIn('Spoken Command: "successful command"', output)

    def test_empty_stream(self):
        pcm, output = self.run_stream([b''], [], '')
        self.assertEqual(pcm, b'')
        self.assertIn('Spoken Command: ""', output)


if __name__ == '__main__':
    unittest.main()
