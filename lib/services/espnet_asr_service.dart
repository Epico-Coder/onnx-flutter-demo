import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:yaml/yaml.dart';

import '../audio/log_mel_extractor.dart';
import '../audio/wav_reader.dart';
import '../debug/intermediate_feature_writer.dart';
import '../decoding/ctc_decoder.dart';
import '../decoding/ctc_prefix_beam_search.dart';
import '../decoding/joint_ctc_transformer_beam_search.dart';
import '../decoding/transformer_decoder_runner.dart';

enum DecodingMode {
  greedyCtc,
  ctcPrefixBeam,
  jointCtcTransformerBeam,
}

extension DecodingModeLabel on DecodingMode {
  String get label {
    switch (this) {
      case DecodingMode.greedyCtc:
        return 'Greedy CTC';
      case DecodingMode.ctcPrefixBeam:
        return 'CTC Prefix Beam';
      case DecodingMode.jointCtcTransformerBeam:
        return 'Joint CTC + Transformer';
    }
  }
}

class EspnetAsrService {
  static const _modelAssetDir = 'assets/models/espnet_onnx';

  final OnnxRuntime _ort = OnnxRuntime();

  dynamic _encoderSession;
  dynamic _ctcSession;
  dynamic _decoderSession;

  bool _ready = false;

  List<String> _tokens = [];

  Future<void> init() async {
    final modelDir = await _copyModelAssetsToStorage();

    _encoderSession = await _ort.createSession(
      '$modelDir/default_encoder.onnx',
    );

    _ctcSession = await _ort.createSession('$modelDir/ctc.onnx');

    // Drop CORE_ML if you see CoreMLExecutionProvider fallback ping-pong.
    final decoderProviders = Platform.isMacOS || Platform.isIOS
        ? [OrtProvider.CORE_ML]
        : <OrtProvider>[];
    _decoderSession = await _ort.createSession(
      '$modelDir/xformer_decoder.onnx',
      options: decoderProviders.isEmpty
          ? null
          : OrtSessionOptions(providers: decoderProviders),
    );

    _tokens = await _loadTokensFromFile('$modelDir/config.yaml');

    _printModelIoNames();

    _ready = true;
  }

  Future<String> transcribeWav(
    String wavPath, {
    DecodingMode decodingMode = DecodingMode.ctcPrefixBeam,
  }) async {
    if (!_ready) {
      throw StateError('ASR service is not ready yet');
    }

    final samples = WavReader.readMonoPcm16AsFloat32(wavPath);
    final artifactDir = await _createIntermediateFeatureRunDir(wavPath);

    final extractor = LogMelExtractor(
      sampleRate: 16000,
      nFft: 512,
      winLength: 400,
      hopLength: 160,
      nMels: 80,
    );

    final extraction = extractor.extractWithIntermediates(samples);
    final features = extraction.normalizedFeatures;
    final numFrames = features.length ~/ 80;

    await IntermediateFeatureWriter.writePreprocessing(
      directory: artifactDir,
      resampledWaveform: samples,
      extraction: extraction,
    );

    final featsTensor = await OrtValue.fromList(features, [1, numFrames, 80]);

    final encoderInputs = <String, OrtValue>{
      _encoderSession.inputNames[0]: featsTensor,
    };

    if (_encoderSession.inputNames.length > 1) {
      encoderInputs[_encoderSession.inputNames[1]] = await OrtValue.fromList(
        Int64List.fromList([numFrames]),
        [1],
      );
    }

    final encoderOutputs = await _encoderSession.run(encoderInputs);

    final encoderOutName = _encoderSession.outputNames[0];

    final encoderOutLensName = _encoderSession.outputNames.length > 1
        ? _encoderSession.outputNames[1]
        : null;

    final encoderOut = encoderOutputs[encoderOutName];
    final encoderOutput = await encoderOut.asFlattenedList();

    final ctcInputs = <String, OrtValue>{_ctcSession.inputNames[0]: encoderOut};

    if (_ctcSession.inputNames.length > 1 && encoderOutLensName != null) {
      ctcInputs[_ctcSession.inputNames[1]] = encoderOutputs[encoderOutLensName];
    }

    final ctcOutputs = await _ctcSession.run(ctcInputs);

    final ctcOutputName = _ctcSession.outputNames[0];
    final ctcTensor = ctcOutputs[ctcOutputName];

    final logits = await ctcTensor.asFlattenedList();
    final shape = ctcTensor.shape;

    final ctcLogits = logits.cast<double>();
    final ctcShape = shape.cast<int>();
    final tokenIds = await _decode(
      encoderOut: encoderOut,
      ctcLogits: ctcLogits,
      ctcShape: ctcShape,
      decodingMode: decodingMode,
    );

    await IntermediateFeatureWriter.writeInference(
      directory: artifactDir,
      encoderOutput: encoderOutput.cast<double>(),
      encoderOutputShape: encoderOut.shape.cast<int>(),
      ctcOutput: ctcLogits,
      ctcOutputShape: ctcShape,
      tokenIds: tokenIds,
    );
    debugPrint('Wrote intermediate ASR features to: ${artifactDir.path}');

    return CtcDecoder.tokensToText(tokenIds, _tokens);
  }

  Future<List<int>> _decode({
    required dynamic encoderOut,
    required List<double> ctcLogits,
    required List<int> ctcShape,
    required DecodingMode decodingMode,
  }) async {
    switch (decodingMode) {
      case DecodingMode.greedyCtc:
        return CtcDecoder.greedyFromLogits(
          ctcLogits,
          shape: ctcShape,
          blankId: 0,
          eosId: 4999,
        );
      case DecodingMode.ctcPrefixBeam:
        return CtcPrefixBeamSearch.decode(
          ctcLogits,
          shape: ctcShape,
          blankId: 0,
          eosId: 4999,
          beamSize: 20,
        );
      case DecodingMode.jointCtcTransformerBeam:
        final runner = OrtTransformerDecoderRunner(
          decoderSession: _decoderSession,
          encoderOut: encoderOut,
          vocab: ctcShape.last,
        );
        return JointCtcTransformerBeamSearch(
          decoder: runner,
          ctcLogits: ctcLogits,
          ctcShape: ctcShape,
          blankId: 0,
          sosId: 4999,
          eosId: 4999,
          beamSize: 10,
          tokenPruneSize: 25,
          ctcWeight: 0.3,
          decoderWeight: 0.7,
        ).decode();
    }
  }

  Future<Directory> _createIntermediateFeatureRunDir(String wavPath) async {
    final documents = await getApplicationDocumentsDirectory();

    return IntermediateFeatureWriter.createRunDirectory(
      rootPath: '${documents.path}/intermediate_features',
      wavPath: wavPath,
    );
  }

  Future<String> _copyModelAssetsToStorage() async {
    final documents = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${documents.path}/espnet_onnx');

    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }

    const assetPaths = [
      '$_modelAssetDir/config.yaml',
      '$_modelAssetDir/bpe.model',
      '$_modelAssetDir/default_encoder.onnx',
      '$_modelAssetDir/ctc.onnx',
      '$_modelAssetDir/xformer_decoder.onnx',
    ];

    final expectedFilenames =
        assetPaths.map((p) => p.split('/').last).toSet();

    // ONNX runtime silently loads `*.onnx.data` sidecars next to a `.onnx`,
    // so a leftover sidecar from a previous export would quietly swap in
    // the old weights.
    for (final entity in modelDir.listSync()) {
      if (entity is File) {
        final name = entity.uri.pathSegments.last;
        if (!expectedFilenames.contains(name)) {
          await entity.delete();
        }
      }
    }

    for (final assetPath in assetPaths) {
      final filename = assetPath.split('/').last;
      final outputFile = File('${modelDir.path}/$filename');

      final bundled = await rootBundle.load(assetPath);
      final bundledSize = bundled.lengthInBytes;
      final shouldOverwrite = !await outputFile.exists() ||
          await outputFile.length() != bundledSize;

      if (shouldOverwrite) {
        await outputFile.writeAsBytes(
          bundled.buffer.asUint8List(bundled.offsetInBytes, bundledSize),
          flush: true,
        );
      }
    }

    return modelDir.path;
  }

  Future<List<String>> _loadTokensFromFile(String configPath) async {
    final text = await File(configPath).readAsString();
    final yaml = loadYaml(text);

    final list = yaml['token']['list'] as YamlList;

    return list.map((e) => e.toString()).toList();
  }

  void _printModelIoNames() {
    debugPrint('Encoder inputs: ${_encoderSession.inputNames}');
    debugPrint('Encoder outputs: ${_encoderSession.outputNames}');
    debugPrint('CTC inputs: ${_ctcSession.inputNames}');
    debugPrint('CTC outputs: ${_ctcSession.outputNames}');
    debugPrint('Decoder inputs: ${_decoderSession.inputNames}');
    debugPrint('Decoder outputs: ${_decoderSession.outputNames}');
  }

  void dispose() {}
}
