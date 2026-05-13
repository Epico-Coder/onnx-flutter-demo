import 'dart:math';
import 'dart:typed_data';

import 'ctc_logits.dart';
import 'transformer_decoder_runner.dart';

class JointCtcTransformerBeamSearch {
  final TransformerDecoderRunner decoder;
  final List<double> ctcLogits;
  final List<int> ctcShape;
  final int blankId;
  final int sosId;
  final int eosId;
  final int beamSize;
  final int tokenPruneSize;
  final double ctcWeight;
  final double decoderWeight;

  late final _CtcLogProbs _ctcLogProbs = _CtcLogProbs(ctcLogits, ctcShape);
  final Map<String, double> _ctcPrefixScoreCache = {};

  JointCtcTransformerBeamSearch({
    required this.decoder,
    required this.ctcLogits,
    required this.ctcShape,
    required this.blankId,
    required this.sosId,
    required this.eosId,
    this.beamSize = 20,
    this.tokenPruneSize = 40,
    this.ctcWeight = 0.3,
    this.decoderWeight = 0.7,
  });

  Future<List<int>> decode() async {
    final initialCaches = await decoder.initialCaches();

    var active = <_JointHypothesis>[
      _JointHypothesis(
        yseq: [sosId],
        tokens: const [],
        tokensKey: '',
        decoderScore: 0.0,
        ctcScore: _ctcPrefixScore(const [], ''),
        caches: initialCaches,
      ),
    ];
    final ended = <_JointHypothesis>[];
    final maxOutputLength = max(1, _ctcLogProbs.time);

    for (int step = 0; step < maxOutputLength; step++) {
      final isFinalStep = step == maxOutputLength - 1;
      final candidates = <_JointHypothesis>[];

      for (final hyp in active) {
        final result = await decoder.step(
          prefix: hyp.yseq,
          caches: hyp.caches,
        );
        final logProbs = result.logProbs;
        final nextCaches = result.caches;

        if (isFinalStep) {
          // Force-close every still-active hypothesis at maxlen so the search
          // always has ended candidates to choose from, matching espnet's
          // `post_process` final-step behaviour.
          candidates.add(
            _JointHypothesis(
              yseq: [...hyp.yseq, eosId],
              tokens: hyp.tokens,
              tokensKey: hyp.tokensKey,
              decoderScore: hyp.decoderScore + logProbs[eosId],
              ctcScore: hyp.ctcScore,
              caches: nextCaches,
              ended: true,
            ),
          );
          continue;
        }

        final topIds = LogMath.topTokenIdsAtFrame(
          logProbs,
          0,
          logProbs.length,
          tokenPruneSize,
        );
        bool sawEos = false;

        void addCandidate(int tokenId) {
          if (tokenId == blankId) return;
          if (tokenId == sosId && sosId != eosId) return;

          final nextYseq = [...hyp.yseq, tokenId];

          if (tokenId == eosId) {
            sawEos = true;
            candidates.add(
              _JointHypothesis(
                yseq: nextYseq,
                tokens: hyp.tokens,
                tokensKey: hyp.tokensKey,
                decoderScore: hyp.decoderScore + logProbs[tokenId],
                ctcScore: hyp.ctcScore,
                caches: nextCaches,
                ended: true,
              ),
            );
            return;
          }

          final nextTokens = [...hyp.tokens, tokenId];
          final nextKey =
              hyp.tokensKey.isEmpty ? '$tokenId' : '${hyp.tokensKey},$tokenId';
          candidates.add(
            _JointHypothesis(
              yseq: nextYseq,
              tokens: nextTokens,
              tokensKey: nextKey,
              decoderScore: hyp.decoderScore + logProbs[tokenId],
              ctcScore: _ctcPrefixScore(nextTokens, nextKey),
              caches: nextCaches,
            ),
          );
        }

        for (int i = 0; i < topIds.length; i++) {
          addCandidate(topIds[i]);
        }
        if (!sawEos && eosId >= 0 && eosId < logProbs.length) {
          addCandidate(eosId);
        }
      }

      candidates.sort((a, b) => _score(b).compareTo(_score(a)));
      ended.addAll(candidates.where((hyp) => hyp.ended));
      active = candidates.where((hyp) => !hyp.ended).take(beamSize).toList();

      if (active.isEmpty) break;

      // Espnet-style end detection: stop when the best ended hyp at the last
      // three lengths is at least D_end (10 nats) worse than the global best
      // ended hyp — i.e. ended-hyp quality has plateaued and longer searches
      // won't improve the result.
      if (_endDetect(ended, step)) break;
    }

    final all = [...ended, ...active]
      ..sort((a, b) => _score(b).compareTo(_score(a)));

    return all.first.tokens;
  }

  double _score(_JointHypothesis hyp) {
    if (hyp.ctcScore == LogMath.logZero) {
      return decoderWeight * hyp.decoderScore;
    }

    return decoderWeight * hyp.decoderScore + ctcWeight * hyp.ctcScore;
  }

  /// Espnet-equivalent `end_detect`: return true when ended-hyp scores have
  /// stagnated across the last `M` step lengths, i.e. growing the beam any
  /// further is very unlikely to find a better-scoring complete hypothesis.
  bool _endDetect(List<_JointHypothesis> endedHyps, int step,
      {int m = 3, double dEnd = -10.0}) {
    if (endedHyps.isEmpty) return false;

    double bestScore = double.negativeInfinity;
    for (final h in endedHyps) {
      final s = _score(h);
      if (s > bestScore) bestScore = s;
    }

    int count = 0;
    for (int offset = 0; offset < m; offset++) {
      final yseqLength = step + 1 - offset;
      if (yseqLength <= 0) break;

      double bestAtLen = double.negativeInfinity;
      for (final h in endedHyps) {
        if (h.yseq.length != yseqLength) continue;
        final s = _score(h);
        if (s > bestAtLen) bestAtLen = s;
      }
      if (bestAtLen == double.negativeInfinity) continue;

      if (bestAtLen - bestScore < dEnd) count++;
    }

    return count == m;
  }

  double _ctcPrefixScore(List<int> prefix, String key) {
    final cached = _ctcPrefixScoreCache[key];
    if (cached != null) return cached;

    final score = _computeCtcPrefixScore(prefix);
    _ctcPrefixScoreCache[key] = score;
    return score;
  }

  double _computeCtcPrefixScore(List<int> prefix) {
    if (prefix.isEmpty) {
      var score = 0.0;

      for (int t = 0; t < _ctcLogProbs.time; t++) {
        score += _ctcLogProbs.at(t, blankId);
      }

      return score;
    }

    final labels = <int>[];

    for (final id in prefix) {
      labels
        ..add(blankId)
        ..add(id);
    }

    labels.add(blankId);

    var previous = List<double>.filled(labels.length, LogMath.logZero);
    previous[0] = _ctcLogProbs.at(0, blankId);

    if (labels.length > 1) {
      previous[1] = _ctcLogProbs.at(0, labels[1]);
    }

    for (int t = 1; t < _ctcLogProbs.time; t++) {
      final current = List<double>.filled(labels.length, LogMath.logZero);

      for (int s = 0; s < labels.length; s++) {
        var total = previous[s];

        if (s > 0) {
          total = LogMath.logAdd(total, previous[s - 1]);
        }

        if (s > 1 && labels[s] != blankId && labels[s] != labels[s - 2]) {
          total = LogMath.logAdd(total, previous[s - 2]);
        }

        current[s] = total + _ctcLogProbs.at(t, labels[s]);
      }

      previous = current;
    }

    return LogMath.logAdd(
      previous[labels.length - 1],
      previous[labels.length - 2],
    );
  }
}

class _CtcLogProbs {
  final List<double> logits;
  final int time;
  final int vocab;
  late final Float64List _logProbs;

  _CtcLogProbs(this.logits, List<int> shape)
      : time = _timeFromShape(shape),
        vocab = _vocabFromShape(shape) {
    final layout = CtcLogitsLayout(time: time, vocab: vocab);
    _logProbs = LogMath.logSoftmaxAllFrames(logits, layout);
  }

  double at(int timeIndex, int vocabIndex) {
    return _logProbs[timeIndex * vocab + vocabIndex];
  }

  static int _timeFromShape(List<int> shape) {
    if (shape.length == 2) return shape[0];
    if (shape.length == 3 && shape[0] == 1) return shape[1];
    if (shape.length == 3 && shape[1] == 1) return shape[0];
    throw ArgumentError('Unsupported CTC logits shape: $shape');
  }

  static int _vocabFromShape(List<int> shape) {
    if (shape.length == 2) return shape[1];
    if (shape.length == 3) return shape[2];
    throw ArgumentError('Unsupported CTC logits shape: $shape');
  }
}

class _JointHypothesis {
  final List<int> yseq;
  final List<int> tokens;
  final String tokensKey;
  final double decoderScore;
  final double ctcScore;
  final List<Object> caches;
  final bool ended;

  const _JointHypothesis({
    required this.yseq,
    required this.tokens,
    required this.tokensKey,
    required this.decoderScore,
    required this.ctcScore,
    required this.caches,
    this.ended = false,
  });
}
