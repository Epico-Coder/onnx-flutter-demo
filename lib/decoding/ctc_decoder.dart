import 'ctc_logits.dart';

class CtcDecoder {
  static List<int> greedyFromLogits(
    List<double> logits, {
    required List<int> shape,
    required int blankId,
    required int eosId,
  }) {
    final layout = CtcLogitsLayout.fromShape(shape);
    final time = layout.time;
    final vocab = layout.vocab;
    final ids = <int>[];
    int previous = -1;

    for (int t = 0; t < time; t++) {
      final base = t * vocab;
      int bestId = 0;
      double bestValue = double.negativeInfinity;

      for (int v = 0; v < vocab; v++) {
        final value = logits[base + v];
        if (value > bestValue) {
          bestValue = value;
          bestId = v;
        }
      }

      if (bestId != blankId && bestId != previous && bestId != eosId) {
        ids.add(bestId);
      }
      previous = bestId;
    }

    return ids;
  }

  static String tokensToText(List<int> ids, List<String> tokens) {
    final buf = StringBuffer();

    for (final id in ids) {
      if (id < 0 || id >= tokens.length) continue;
      final token = tokens[id];
      if (token.startsWith('<') && token.endsWith('>')) continue;
      buf.write(token);
    }

    return buf
        .toString()
        .replaceAll('▁', ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }
}
