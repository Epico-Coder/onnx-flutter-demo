import 'dart:math';

class CtcLogitsLayout {
  final int time;
  final int vocab;
  final int Function(int timeIndex, int vocabIndex) index;

  const CtcLogitsLayout({
    required this.time,
    required this.vocab,
    required this.index,
  });

  factory CtcLogitsLayout.fromShape(List<int> shape) {
    if (shape.length == 2) {
      final time = shape[0];
      final vocab = shape[1];
      return CtcLogitsLayout(
        time: time,
        vocab: vocab,
        index: (t, v) => t * vocab + v,
      );
    }

    if (shape.length == 3 && shape[0] == 1) {
      final time = shape[1];
      final vocab = shape[2];
      return CtcLogitsLayout(
        time: time,
        vocab: vocab,
        index: (t, v) => t * vocab + v,
      );
    }

    if (shape.length == 3 && shape[1] == 1) {
      final time = shape[0];
      final vocab = shape[2];
      return CtcLogitsLayout(
        time: time,
        vocab: vocab,
        index: (t, v) => t * vocab + v,
      );
    }

    throw ArgumentError('Unsupported CTC logits shape: $shape');
  }
}

class LogMath {
  static const double logZero = double.negativeInfinity;

  /// log(exp(a) + exp(b)) computed without overflow.
  static double logAdd(double a, double b) {
    if (a == logZero) return b;
    if (b == logZero) return a;

    final larger = max(a, b);
    final smaller = min(a, b);

    return larger + log(1.0 + exp(smaller - larger));
  }

  static List<double> frameLogSoftmax(
    List<double> logits,
    CtcLogitsLayout layout,
    int timeIndex,
  ) {
    final values = List<double>.filled(layout.vocab, 0.0);
    var maxValue = double.negativeInfinity;

    for (int v = 0; v < layout.vocab; v++) {
      final value = logits[layout.index(timeIndex, v)];
      values[v] = value;
      if (value > maxValue) maxValue = value;
    }

    var sumExp = 0.0;
    for (final value in values) {
      sumExp += exp(value - maxValue);
    }

    final logSumExp = maxValue + log(sumExp);

    for (int v = 0; v < values.length; v++) {
      values[v] -= logSumExp;
    }

    return values;
  }

  static List<int> topTokenIds(List<double> logProbs, int count) {
    final limit = min(count, logProbs.length);
    final ids = <int>[];

    for (int candidate = 0; candidate < logProbs.length; candidate++) {
      var insertAt = ids.length;

      while (insertAt > 0 &&
          logProbs[candidate] > logProbs[ids[insertAt - 1]]) {
        insertAt--;
      }

      if (insertAt < limit) {
        ids.insert(insertAt, candidate);
        if (ids.length > limit) {
          ids.removeLast();
        }
      }
    }

    return ids;
  }
}
