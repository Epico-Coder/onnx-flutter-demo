import 'ctc_logits.dart';

class CtcPrefixBeamSearch {
  static List<int> decode(
    List<double> logits, {
    required List<int> shape,
    required int blankId,
    required int eosId,
    int beamSize = 20,
    int tokenPruneSize = 40,
    Set<int> suppressedTokenIds = const {},
  }) {
    final layout = CtcLogitsLayout.fromShape(shape);

    if (blankId < 0 || blankId >= layout.vocab) {
      throw ArgumentError(
        'blankId $blankId is outside vocab size ${layout.vocab}',
      );
    }

    var beam = <_Prefix, _BeamState>{
      const _Prefix(<int>[]): const _BeamState(
        blank: 0.0,
        nonBlank: LogMath.logZero,
      ),
    };

    for (int t = 0; t < layout.time; t++) {
      final frame = LogMath.frameLogSoftmax(logits, layout, t);
      final topTokens = LogMath.topTokenIds(frame, tokenPruneSize);
      final next = <_Prefix, _BeamState>{};

      for (final entry in beam.entries) {
        final prefix = entry.key;
        final state = entry.value;
        final prefixScore = state.score;

        _merge(
          next,
          prefix,
          blank: prefixScore + frame[blankId],
          nonBlank: LogMath.logZero,
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
              blank: LogMath.logZero,
              nonBlank: state.nonBlank + tokenLogProb,
            );

            final extended = prefix.extend(tokenId);
            _merge(
              next,
              extended,
              blank: LogMath.logZero,
              nonBlank: state.blank + tokenLogProb,
            );
          } else {
            final extended = prefix.extend(tokenId);
            _merge(
              next,
              extended,
              blank: LogMath.logZero,
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
      blank: LogMath.logAdd(current.blank, blank),
      nonBlank: LogMath.logAdd(current.nonBlank, nonBlank),
    );
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
    this.blank = LogMath.logZero,
    this.nonBlank = LogMath.logZero,
  });

  double get score => LogMath.logAdd(blank, nonBlank);
}
