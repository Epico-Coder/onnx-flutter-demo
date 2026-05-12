import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:onnx_inference_app/decoding/joint_ctc_transformer_beam_search.dart';
import 'package:onnx_inference_app/decoding/transformer_decoder_runner.dart';

const _sos = 4999;
const _eos = 4999;
const _blank = 0;
const _vocab = 5000;

class _Call {
  final List<int> prefix;
  final List<int> cacheLengths;
  _Call(this.prefix, this.cacheLengths);
}

/// Caches are just integers — the count of times the cache has grown.
/// The runner returns each new cache as `parentCache + 1`, so we can verify
/// caches are threaded across decoding steps.
class _CountingCache {
  final int length;
  const _CountingCache(this.length);
}

class _FakeRunner implements TransformerDecoderRunner {
  final List<_Call> calls = [];
  final int Function(List<int> prefix) bestTokenForPrefix;

  _FakeRunner(this.bestTokenForPrefix);

  @override
  int get vocab => _vocab;

  @override
  int get numLayers => 6;

  @override
  Future<List<Object>> initialCaches() async {
    return List<Object>.generate(numLayers, (_) => const _CountingCache(0));
  }

  @override
  Future<TransformerDecoderStep> step({
    required List<int> prefix,
    required List<Object> caches,
  }) async {
    final lengths =
        caches.map((c) => (c as _CountingCache).length).toList(growable: false);
    calls.add(_Call(List<int>.from(prefix), lengths));

    final lp = Float64List(_vocab);
    for (int i = 0; i < _vocab; i++) {
      lp[i] = -1000.0;
    }
    final best = bestTokenForPrefix(prefix);
    lp[best] = -0.01;

    final next = <Object>[
      for (final c in caches) _CountingCache((c as _CountingCache).length + 1),
    ];

    return TransformerDecoderStep(logProbs: lp, caches: next);
  }
}

/// Build a tiny CTC log-probs grid: one frame, blank dominates so prefix
/// scoring is well-defined but doesn't fight the decoder.
List<double> _ctcLogits({required int frames, required int vocab}) {
  final logits = List<double>.filled(frames * vocab, -1000.0);
  for (int t = 0; t < frames; t++) {
    logits[t * vocab + _blank] = 0.0;
  }
  return logits;
}

void main() {
  test('decoder is called with the full prefix and threaded caches', () async {
    // The fake decoder picks token 7 first, then 42, then EOS — so we expect
    // beam search to walk that path and call the decoder 3 times along it.
    final picks = [7, 42, _eos];
    int call = 0;
    final runner = _FakeRunner((prefix) {
      final idx = call.clamp(0, picks.length - 1);
      call++;
      return picks[idx];
    });

    final tokens = await JointCtcTransformerBeamSearch(
      decoder: runner,
      ctcLogits: _ctcLogits(frames: 8, vocab: _vocab),
      ctcShape: const [1, 8, _vocab],
      blankId: _blank,
      sosId: _sos,
      eosId: _eos,
      beamSize: 1,
      tokenPruneSize: 4,
    ).decode();

    expect(tokens, [7, 42]);

    // Step 0: just <sos>, every cache empty.
    final first = runner.calls.first;
    expect(first.prefix, [_sos]);
    expect(first.cacheLengths, List.filled(6, 0));

    // Step 1: <sos> + 7, every cache is now length 1 (grown by the runner).
    final second = runner.calls[1];
    expect(second.prefix, [_sos, 7]);
    expect(second.cacheLengths, List.filled(6, 1));

    // Step 2: <sos> + 7 + 42, caches length 2.
    final third = runner.calls[2];
    expect(third.prefix, [_sos, 7, 42]);
    expect(third.cacheLengths, List.filled(6, 2));
  });

  test('caches are inherited from the parent hypothesis when the beam splits',
      () async {
    // The decoder splits — prefix [_sos] favors token 7, prefix [_sos, 7]
    // favors EOS. With beam=1, we still expect cache lengths to track the
    // chain depth.
    final runner = _FakeRunner((prefix) {
      if (prefix.length == 1) return 7;
      return _eos;
    });

    await JointCtcTransformerBeamSearch(
      decoder: runner,
      ctcLogits: _ctcLogits(frames: 4, vocab: _vocab),
      ctcShape: const [1, 4, _vocab],
      blankId: _blank,
      sosId: _sos,
      eosId: _eos,
      beamSize: 1,
      tokenPruneSize: 2,
    ).decode();

    // Both calls walk the same chain, so the second call's cache lengths
    // are exactly one greater than the first's.
    final a = runner.calls[0].cacheLengths;
    final b = runner.calls[1].cacheLengths;
    for (int i = 0; i < a.length; i++) {
      expect(b[i], a[i] + 1);
    }
  });
}
