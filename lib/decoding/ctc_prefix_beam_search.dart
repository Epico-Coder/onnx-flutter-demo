import 'dart:math';

class CtcPrefixBeamSearch {
  static const double _logZero = double.negativeInfinity;

  static List<int> decode(
    List<double> logits, {
    required List<int> shape,
    required int blankId,
    required int eosId,
    int beamSize = 20,
    int tokenPruneSize = 40,
    Set<int> suppressedTokenIds = const {},
  }) {
    final layout = _LogitLayout.fromShape(shape);
    final vocab = layout.vocab;

    if (blankId < 0 || blankId >= vocab) {
      throw ArgumentError('blankId $blankId is outside vocab size $vocab');
    }

    var beam = <_Prefix, _BeamState>{
      const _Prefix(<int>[]): const _BeamState(blank: 0.0, nonBlank: _logZero),
    };

    for (int t = 0; t < layout.time; t++) {
      final frame = _frameLogProbs(logits, layout, t);
      final topTokens = _topTokenIds(frame, tokenPruneSize);
      final next = <_Prefix, _BeamState>{};

      for (final entry in beam.entries) {
        final prefix = entry.key;
        final state = entry.value;
        final prefixScore = state.score;

        _merge(
          next,
          prefix,
          blank: prefixScore + frame[blankId],
          nonBlank: _logZero,
        );

        for (final tokenId in topTokens) {
          if (tokenId == blankId) continue;
          if (suppressedTokenIds.contains(tokenId)) continue;

          final previousToken = prefix.lastOrNull;
          final tokenLogProb = frame[tokenId];

          if (tokenId == previousToken) {
            _merge(
              next,
              prefix,
              blank: _logZero,
              nonBlank: state.nonBlank + tokenLogProb,
            );

            final extended = prefix.extend(tokenId);
            _merge(
              next,
              extended,
              blank: _logZero,
              nonBlank: state.blank + tokenLogProb,
            );
          } else {
            final extended = prefix.extend(tokenId);
            _merge(
              next,
              extended,
              blank: _logZero,
              nonBlank: prefixScore + tokenLogProb,
            );
          }
        }
      }

      beam = _pruneBeam(next, beamSize);
    }

    final best = beam.entries.reduce(
      (a, b) => a.value.score >= b.value.score ? a : b,
    );

    return best.key.ids.where((id) => id != eosId).toList();
  }

  static List<double> _frameLogProbs(
    List<double> logits,
    _LogitLayout layout,
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

    // ESPnet CTC exports commonly emit log-softmax already. If the frame is
    // already normalized, this subtraction is effectively a no-op.
    for (int v = 0; v < values.length; v++) {
      values[v] -= logSumExp;
    }

    return values;
  }

  static List<int> _topTokenIds(List<double> logProbs, int count) {
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

  static Map<_Prefix, _BeamState> _pruneBeam(
    Map<_Prefix, _BeamState> beam,
    int beamSize,
  ) {
    final entries = beam.entries.toList()
      ..sort((a, b) => b.value.score.compareTo(a.value.score));

    return Map<_Prefix, _BeamState>.fromEntries(entries.take(beamSize));
  }

  static void _merge(
    Map<_Prefix, _BeamState> beam,
    _Prefix prefix, {
    required double blank,
    required double nonBlank,
  }) {
    final current = beam[prefix] ?? const _BeamState();

    beam[prefix] = _BeamState(
      blank: _logAdd(current.blank, blank),
      nonBlank: _logAdd(current.nonBlank, nonBlank),
    );
  }

  static double _logAdd(double a, double b) {
    if (a == _logZero) return b;
    if (b == _logZero) return a;

    final larger = max(a, b);
    final smaller = min(a, b);

    return larger + log(1.0 + exp(smaller - larger));
  }
}

class _LogitLayout {
  final int time;
  final int vocab;
  final int Function(int timeIndex, int vocabIndex) index;

  const _LogitLayout({
    required this.time,
    required this.vocab,
    required this.index,
  });

  factory _LogitLayout.fromShape(List<int> shape) {
    if (shape.length == 2) {
      final time = shape[0];
      final vocab = shape[1];

      return _LogitLayout(
        time: time,
        vocab: vocab,
        index: (t, v) => t * vocab + v,
      );
    }

    if (shape.length == 3 && shape[0] == 1) {
      final time = shape[1];
      final vocab = shape[2];

      return _LogitLayout(
        time: time,
        vocab: vocab,
        index: (t, v) => t * vocab + v,
      );
    }

    if (shape.length == 3 && shape[1] == 1) {
      final time = shape[0];
      final vocab = shape[2];

      return _LogitLayout(
        time: time,
        vocab: vocab,
        index: (t, v) => t * vocab + v,
      );
    }

    throw ArgumentError('Unsupported CTC logits shape: $shape');
  }
}

class _Prefix {
  final List<int> ids;

  const _Prefix(this.ids);

  int? get lastOrNull => ids.isEmpty ? null : ids.last;

  _Prefix extend(int id) => _Prefix([...ids, id]);

  @override
  bool operator ==(Object other) {
    if (other is! _Prefix || ids.length != other.ids.length) return false;

    for (int i = 0; i < ids.length; i++) {
      if (ids[i] != other.ids[i]) return false;
    }

    return true;
  }

  @override
  int get hashCode => Object.hashAll(ids);
}

class _BeamState {
  final double blank;
  final double nonBlank;

  const _BeamState({
    this.blank = CtcPrefixBeamSearch._logZero,
    this.nonBlank = CtcPrefixBeamSearch._logZero,
  });

  double get score => CtcPrefixBeamSearch._logAdd(blank, nonBlank);
}
