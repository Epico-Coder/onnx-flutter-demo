import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnx_inference_app/decoding/ctc_decoder.dart';
import 'package:onnx_inference_app/decoding/ctc_prefix_beam_search.dart';

void main() {
  final fixture = _loadCtcFixture('test/ctc_log_probs.bin');

  test('greedy timing', () {
    // Warm up the JIT.
    for (int i = 0; i < 5; i++) {
      CtcDecoder.greedyFromLogits(
        fixture.logits,
        shape: fixture.shape,
        blankId: 0,
        eosId: 4999,
      );
    }

    final sw = Stopwatch()..start();
    const n = 50;
    for (int i = 0; i < n; i++) {
      CtcDecoder.greedyFromLogits(
        fixture.logits,
        shape: fixture.shape,
        blankId: 0,
        eosId: 4999,
      );
    }
    sw.stop();
    print('greedy avg: ${(sw.elapsedMicroseconds / n).toStringAsFixed(1)} us');
  });

  test('prefix beam timing', () {
    for (int i = 0; i < 3; i++) {
      CtcPrefixBeamSearch.decode(
        fixture.logits,
        shape: fixture.shape,
        blankId: 0,
        eosId: 4999,
        beamSize: 20,
        tokenPruneSize: 40,
      );
    }

    final sw = Stopwatch()..start();
    const n = 20;
    for (int i = 0; i < n; i++) {
      CtcPrefixBeamSearch.decode(
        fixture.logits,
        shape: fixture.shape,
        blankId: 0,
        eosId: 4999,
        beamSize: 20,
        tokenPruneSize: 40,
      );
    }
    sw.stop();
    print(
      'prefix beam avg: ${(sw.elapsedMicroseconds / n / 1000).toStringAsFixed(2)} ms',
    );
  });
}

class _CtcFixture {
  final List<double> logits;
  final List<int> shape;
  _CtcFixture(this.logits, this.shape);
}

_CtcFixture _loadCtcFixture(String path) {
  final bytes = File(path).readAsBytesSync();
  final view = ByteData.sublistView(bytes);
  final b = view.getInt32(0, Endian.little);
  final t = view.getInt32(4, Endian.little);
  final v = view.getInt32(8, Endian.little);

  final floatCount = b * t * v;
  final floats = Float32List.view(
    bytes.buffer,
    bytes.offsetInBytes + 12,
    floatCount,
  );

  return _CtcFixture(List<double>.from(floats), [b, t, v]);
}
