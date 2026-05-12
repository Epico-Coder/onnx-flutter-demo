import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnx_inference_app/audio/log_mel_extractor.dart';
import 'package:onnx_inference_app/audio/wav_reader.dart';

void main() {
  test('Dart log-mel features match librosa within tolerance', () {
    final samples = WavReader.readMonoPcm16AsFloat32('assets/test.wav');

    final extractor = LogMelExtractor(
      sampleRate: 16000,
      nFft: 512,
      winLength: 400,
      hopLength: 160,
      nMels: 80,
    );

    final dartFeats = extractor.extract(samples);

    final reference = _loadReference('test/log_mel.bin');

    expect(
      dartFeats.length,
      reference.values.length,
      reason: 'frame count mismatch (T=${reference.t}, M=${reference.m})',
    );

    // Aggregate stats are usually much more sensitive to algorithmic drift
    // than spot-checks, so look at mean / max absolute error across the whole
    // utterance.
    double sumAbs = 0.0;
    double maxAbs = 0.0;
    for (int i = 0; i < dartFeats.length; i++) {
      final diff = (dartFeats[i] - reference.values[i]).abs();
      sumAbs += diff;
      if (diff > maxAbs) maxAbs = diff;
    }
    final meanAbs = sumAbs / dartFeats.length;

    print(
      'log-mel parity: meanAbsErr=${meanAbs.toStringAsFixed(4)} '
      'maxAbsErr=${maxAbs.toStringAsFixed(4)} (T=${reference.t}, M=${reference.m})',
    );

    expect(meanAbs, lessThan(1e-4));
    expect(maxAbs, lessThan(1e-3));
  });
}

class _MelReference {
  final int t;
  final int m;
  final Float32List values;
  _MelReference(this.t, this.m, this.values);
}

_MelReference _loadReference(String path) {
  final bytes = File(path).readAsBytesSync();
  final view = ByteData.sublistView(bytes);
  final t = view.getInt32(0, Endian.little);
  final m = view.getInt32(4, Endian.little);
  final values = Float32List.view(
    bytes.buffer,
    bytes.offsetInBytes + 8,
    t * m,
  );
  return _MelReference(t, m, values);
}
