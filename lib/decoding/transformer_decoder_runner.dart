import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class TransformerDecoderStep {
  final Float64List logProbs;
  final List<Object> caches;

  const TransformerDecoderStep({required this.logProbs, required this.caches});
}

abstract class TransformerDecoderRunner {
  int get vocab;
  int get numLayers;

  Future<List<Object>> initialCaches();

  Future<TransformerDecoderStep> step({
    required List<int> prefix,
    required List<Object> caches,
  });
}

class OrtTransformerDecoderRunner implements TransformerDecoderRunner {
  final dynamic decoderSession;
  final OrtValue encoderOut;

  @override
  final int vocab;

  @override
  final int numLayers;

  final int decoderOutputSize;

  late final List<String> _inputNames =
      decoderSession.inputNames.cast<String>();
  late final List<String> _outputNames =
      decoderSession.outputNames.cast<String>();

  OrtTransformerDecoderRunner({
    required this.decoderSession,
    required this.encoderOut,
    required this.vocab,
    this.numLayers = 6,
    this.decoderOutputSize = 256,
  });

  @override
  Future<List<Object>> initialCaches() async {
    final empty = Float32List(0);
    final caches = <Object>[];

    for (int i = 0; i < numLayers; i++) {
      caches.add(
        await OrtValue.fromList(empty, [1, 0, decoderOutputSize]),
      );
    }

    return caches;
  }

  @override
  Future<TransformerDecoderStep> step({
    required List<int> prefix,
    required List<Object> caches,
  }) async {
    final tgtTensor = await OrtValue.fromList(
      Int64List.fromList(prefix),
      [1, prefix.length],
    );

    final inputs = <String, OrtValue>{
      _inputNames[0]: tgtTensor,
      _inputNames[1]: encoderOut,
    };

    for (int i = 0; i < numLayers; i++) {
      inputs[_inputNames[i + 2]] = caches[i] as OrtValue;
    }

    final outputs = await decoderSession.run(inputs);

    final logSoftmax = outputs[_outputNames[0]];
    final rawLogProbs = await logSoftmax.asFlattenedList();
    final logProbs = Float64List(rawLogProbs.length);
    for (int i = 0; i < rawLogProbs.length; i++) {
      logProbs[i] = (rawLogProbs[i] as num).toDouble();
    }

    final newCaches = <Object>[];
    for (int i = 0; i < numLayers; i++) {
      newCaches.add(outputs[_outputNames[i + 1]] as Object);
    }

    return TransformerDecoderStep(logProbs: logProbs, caches: newCaches);
  }
}
