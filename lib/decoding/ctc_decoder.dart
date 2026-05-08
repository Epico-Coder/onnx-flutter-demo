import 'ctc_logits.dart';

class CtcDecoder {
  static List<int> greedyFromLogits(
    List<double> logits, {
    required List<int> shape,
    required int blankId,
    required int eosId,
  }) {
    final layout = CtcLogitsLayout.fromShape(shape);
    final ids = <int>[];
    int? previous;

    for (int t = 0; t < layout.time; t++) {
      int bestId = 0;
      double bestValue = double.negativeInfinity;

      for (int v = 0; v < layout.vocab; v++) {
        final value = logits[layout.index(t, v)];
        if (value > bestValue) {
          bestValue = value;
          bestId = v;
        }
      }

      final isBlank = bestId == blankId;
      final isRepeat = bestId == previous;
      final isEos = bestId == eosId;

      if (!isBlank && !isRepeat && !isEos) {
        ids.add(bestId);
      }
      previous = bestId;
    }

    return ids;
  }

  static String tokensToText(List<int> ids, List<String> tokens) {
    final pieces = <String>[];

    for (final id in ids) {
      if (id < 0 || id >= tokens.length) continue;
      final token = tokens[id];
      if (token.startsWith('<') && token.endsWith('>')) {
        continue;
      }
      pieces.add(token);
    }

    return pieces
        .join('')
        .replaceAll('\u2581', ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }
}
