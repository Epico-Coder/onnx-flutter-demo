import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnx_inference_app/decoding/ctc_decoder.dart';
import 'package:onnx_inference_app/decoding/ctc_prefix_beam_search.dart';

const _expectedIds = [8, 697, 9, 18, 8, 2960, 15, 1391, 827, 194];
const _expectedText = 'de plant van de aardappel is giftig';

void main() {
  final fixture = _loadCtcFixture('test/ctc_log_probs.bin');

  test('greedy CTC decode matches the Python reference', () {
    final ids = CtcDecoder.greedyFromLogits(
      fixture.logits,
      shape: fixture.shape,
      blankId: 0,
      eosId: 4999,
    );
    expect(ids, _expectedIds);
  });

  test('CTC prefix beam decode matches the Python reference', () {
    final ids = CtcPrefixBeamSearch.decode(
      fixture.logits,
      shape: fixture.shape,
      blankId: 0,
      eosId: 4999,
      beamSize: 20,
      tokenPruneSize: 40,
    );
    expect(ids, _expectedIds);
  });

  test('token rendering produces the expected sentence', () {
    final tokens = _readTokens(
      'assets/models/espnet_onnx/config.yaml',
    );
    final text = CtcDecoder.tokensToText(_expectedIds, tokens);
    expect(text, _expectedText);
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

List<String> _readTokens(String configPath) {
  final lines = File(configPath).readAsLinesSync();
  final tokens = <String>[];
  var inList = false;

  for (final raw in lines) {
    if (!inList) {
      if (raw.trimRight() == '  list:') inList = true;
      continue;
    }

    if (!raw.startsWith('  -')) break;

    final rest = raw.substring(3).trim();
    tokens.add(_unquoteYamlScalar(rest));
  }

  return tokens;
}

String _unquoteYamlScalar(String s) {
  if (s.startsWith("'") && s.endsWith("'") && s.length >= 2) {
    return s.substring(1, s.length - 1).replaceAll("''", "'");
  }

  if (s.startsWith('"') && s.endsWith('"') && s.length >= 2) {
    final inner = s.substring(1, s.length - 1);
    final buf = StringBuffer();
    int i = 0;

    while (i < inner.length) {
      final ch = inner[i];

      if (ch == r'\' && i + 1 < inner.length) {
        final next = inner[i + 1];

        if (next == 'u' && i + 5 < inner.length) {
          final code = int.parse(inner.substring(i + 2, i + 6), radix: 16);
          buf.writeCharCode(code);
          i += 6;
          continue;
        }

        if (next == 'x' && i + 3 < inner.length) {
          final code = int.parse(inner.substring(i + 2, i + 4), radix: 16);
          buf.writeCharCode(code);
          i += 4;
          continue;
        }

        if (next == 'n') {
          buf.write('\n');
          i += 2;
          continue;
        }

        if (next == 't') {
          buf.write('\t');
          i += 2;
          continue;
        }

        if (next == r'\') {
          buf.write(r'\');
          i += 2;
          continue;
        }

        if (next == '"') {
          buf.write('"');
          i += 2;
          continue;
        }
      }

      buf.write(ch);
      i++;
    }

    return buf.toString();
  }

  return s;
}