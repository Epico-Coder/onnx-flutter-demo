class CtcDecoder {
  static List<int> greedyFromLogits(
    List<double> logits, {
    required List<int> shape,
    required int blankId,
    required int eosId,
  }) {
    late final int time;
    late final int vocab;
    late final int frameStride;

    if (shape.length == 3) {
      if (shape[0] == 1) {
        time = shape[1];
        frameStride = shape[2];
      } else if (shape[1] == 1) {
        time = shape[0];
        frameStride = shape[1] * shape[2];
      } else {
        time = shape[1];
        frameStride = shape[2];
      }

      vocab = shape[2];
    } else if (shape.length == 2) {
      time = shape[0];
      vocab = shape[1];
      frameStride = vocab;
    } else {
      throw ArgumentError('Unsupported CTC logits shape: $shape');
    }

    final ids = <int>[];

    int? previous;

    for (int t = 0; t < time; t++) {
      int bestId = 0;
      double bestValue = double.negativeInfinity;

      for (int v = 0; v < vocab; v++) {
        final value = logits[t * frameStride + v];

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
