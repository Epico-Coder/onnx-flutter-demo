import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class JointCtcTransformerBeamSearch {
  static const double _logZero = double.negativeInfinity;

  final dynamic decoderSession;
  final dynamic encoderOut;
  final List<double> ctcLogits;
  final List<int> ctcShape;
  final int blankId;
  final int sosId;
  final int eosId;
  final int beamSize;
  final int tokenPruneSize;
  final double ctcWeight;
  final double decoderWeight;
  final int decoderLayers;
  final int decoderOutputSize;

  late final _CtcLogProbs _ctcLogProbs = _CtcLogProbs(ctcLogits, ctcShape);
  final Map<String, double> _ctcPrefixScoreCache = {};

  JointCtcTransformerBeamSearch({
    required this.decoderSession,
    required this.encoderOut,
    required this.ctcLogits,
    required this.ctcShape,
    required this.blankId,
    required this.sosId,
    required this.eosId,
    this.beamSize = 20,
    this.tokenPruneSize = 40,
    this.ctcWeight = 0.3,
    this.decoderWeight = 0.7,
    this.decoderLayers = 6,
    this.decoderOutputSize = 256,
  });

  Future<List<int>> decode() async {
    var active = <_JointHypothesis>[
      _JointHypothesis(
        yseq: [sosId],
        tokens: const [],
        decoderScore: 0.0,
        ctcScore: _ctcPrefixScore(const []),
        caches: await _initialCaches(),
      ),
    ];
    final ended = <_JointHypothesis>[];
    final maxOutputLength = max(1, _ctcLogProbs.time);

    for (int step = 0; step < maxOutputLength; step++) {
      final candidates = <_JointHypothesis>[];

      for (final hyp in active) {
        final decoderStep = await _runDecoderStep(hyp);
        final logProbs = decoderStep.logProbs;
        final topIds = _topTokenIds(logProbs, tokenPruneSize, includeId: eosId);

        for (final tokenId in topIds) {
          if (tokenId == blankId || tokenId == sosId) continue;

          final nextYseq = [...hyp.yseq, tokenId];

          if (tokenId == eosId) {
            candidates.add(
              hyp.copyWith(
                yseq: nextYseq,
                decoderScore: hyp.decoderScore + logProbs[tokenId],
                caches: decoderStep.caches,
                ended: true,
              ),
            );
            continue;
          }

          final nextTokens = [...hyp.tokens, tokenId];
          candidates.add(
            _JointHypothesis(
              yseq: nextYseq,
              tokens: nextTokens,
              decoderScore: hyp.decoderScore + logProbs[tokenId],
              ctcScore: _ctcPrefixScore(nextTokens),
              caches: decoderStep.caches,
            ),
          );
        }
      }

      candidates.sort((a, b) => _score(b).compareTo(_score(a)));
      ended.addAll(candidates.where((hyp) => hyp.ended));
      active = candidates.where((hyp) => !hyp.ended).take(beamSize).toList();

      if (active.isEmpty) break;

      ended.sort((a, b) => _score(b).compareTo(_score(a)));

      if (ended.length >= beamSize &&
          _score(ended.first) >= _score(active.first)) {
        break;
      }
    }

    final all = [...ended, ...active]
      ..sort((a, b) => _score(b).compareTo(_score(a)));

    return all.first.tokens;
  }

  double _score(_JointHypothesis hyp) {
    if (hyp.ctcScore == _logZero) {
      return decoderWeight * hyp.decoderScore;
    }

    return decoderWeight * hyp.decoderScore + ctcWeight * hyp.ctcScore;
  }

  Future<List<dynamic>> _initialCaches() async {
    final caches = <dynamic>[];

    for (int i = 0; i < decoderLayers; i++) {
      caches.add(await OrtValue.fromList(Float32List(0), [
        1,
        0,
        decoderOutputSize,
      ]));
    }

    return caches;
  }

  Future<_DecoderStep> _runDecoderStep(_JointHypothesis hyp) async {
    final inputs = <String, OrtValue>{};
    final inputNames = decoderSession.inputNames.cast<String>();

    inputs[inputNames[0]] = await OrtValue.fromList(
      Int64List.fromList([hyp.yseq.last]),
      [1, 1],
    );
    inputs[inputNames[1]] = encoderOut;

    for (int i = 0; i < decoderLayers; i++) {
      inputs[inputNames[i + 2]] = hyp.caches[i];
    }

    final outputs = await decoderSession.run(inputs);
    final outputNames = decoderSession.outputNames.cast<String>();
    final logSoftmax = outputs[outputNames[0]];
    final rawLogProbs = (await logSoftmax.asFlattenedList()).cast<double>();
    final logProbs = _normalizeLogProbs(rawLogProbs);
    final caches = <dynamic>[];

    for (int i = 0; i < decoderLayers; i++) {
      caches.add(outputs[outputNames[i + 1]]);
    }

    return _DecoderStep(logProbs, caches);
  }

  List<double> _normalizeLogProbs(List<double> values) {
    var maxValue = double.negativeInfinity;

    for (final value in values) {
      if (value > maxValue) maxValue = value;
    }

    var sumExp = 0.0;

    for (final value in values) {
      sumExp += exp(value - maxValue);
    }

    final logSumExp = maxValue + log(sumExp);

    return values.map((value) => value - logSumExp).toList();
  }

  List<int> _topTokenIds(
    List<double> logProbs,
    int count, {
    required int includeId,
  }) {
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

    if (!ids.contains(includeId) &&
        includeId >= 0 &&
        includeId < logProbs.length) {
      ids.add(includeId);
    }

    return ids;
  }

  double _ctcPrefixScore(List<int> prefix) {
    final key = prefix.join(',');
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

    var previous = List<double>.filled(labels.length, _logZero);
    previous[0] = _ctcLogProbs.at(0, blankId);

    if (labels.length > 1) {
      previous[1] = _ctcLogProbs.at(0, labels[1]);
    }

    for (int t = 1; t < _ctcLogProbs.time; t++) {
      final current = List<double>.filled(labels.length, _logZero);

      for (int s = 0; s < labels.length; s++) {
        var total = previous[s];

        if (s > 0) {
          total = _logAdd(total, previous[s - 1]);
        }

        if (s > 1 && labels[s] != blankId && labels[s] != labels[s - 2]) {
          total = _logAdd(total, previous[s - 2]);
        }

        current[s] = total + _ctcLogProbs.at(t, labels[s]);
      }

      previous = current;
    }

    return _logAdd(previous[labels.length - 1], previous[labels.length - 2]);
  }

  static double _logAdd(double a, double b) {
    if (a == _logZero) return b;
    if (b == _logZero) return a;

    final larger = max(a, b);
    final smaller = min(a, b);

    return larger + log(1.0 + exp(smaller - larger));
  }
}

class _CtcLogProbs {
  final List<double> logits;
  final int time;
  final int vocab;
  final int Function(int timeIndex, int vocabIndex) index;
  late final List<double> _logProbs = _buildLogProbs();

  _CtcLogProbs(this.logits, List<int> shape)
      : time = _timeFromShape(shape),
        vocab = _vocabFromShape(shape),
        index = _indexFromShape(shape);

  double at(int timeIndex, int vocabIndex) {
    return _logProbs[index(timeIndex, vocabIndex)];
  }

  List<double> _buildLogProbs() {
    final values = List<double>.from(logits);

    for (int t = 0; t < time; t++) {
      var maxValue = double.negativeInfinity;

      for (int v = 0; v < vocab; v++) {
        final value = values[index(t, v)];
        if (value > maxValue) maxValue = value;
      }

      var sumExp = 0.0;

      for (int v = 0; v < vocab; v++) {
        sumExp += exp(values[index(t, v)] - maxValue);
      }

      final logSumExp = maxValue + log(sumExp);

      for (int v = 0; v < vocab; v++) {
        final offset = index(t, v);
        values[offset] -= logSumExp;
      }
    }

    return values;
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

  static int Function(int timeIndex, int vocabIndex) _indexFromShape(
    List<int> shape,
  ) {
    final vocab = _vocabFromShape(shape);

    if (shape.length == 2) {
      return (t, v) => t * vocab + v;
    }

    if (shape.length == 3 && shape[0] == 1) {
      return (t, v) => t * vocab + v;
    }

    if (shape.length == 3 && shape[1] == 1) {
      return (t, v) => t * vocab + v;
    }

    throw ArgumentError('Unsupported CTC logits shape: $shape');
  }
}

class _DecoderStep {
  final List<double> logProbs;
  final List<dynamic> caches;

  const _DecoderStep(this.logProbs, this.caches);
}

class _JointHypothesis {
  final List<int> yseq;
  final List<int> tokens;
  final double decoderScore;
  final double ctcScore;
  final List<dynamic> caches;
  final bool ended;

  const _JointHypothesis({
    required this.yseq,
    required this.tokens,
    required this.decoderScore,
    required this.ctcScore,
    required this.caches,
    this.ended = false,
  });

  _JointHypothesis copyWith({
    required List<int> yseq,
    required double decoderScore,
    required List<dynamic> caches,
    required bool ended,
  }) {
    return _JointHypothesis(
      yseq: yseq,
      tokens: tokens,
      decoderScore: decoderScore,
      ctcScore: ctcScore,
      caches: caches,
      ended: ended,
    );
  }
}
