import 'dart:io';
import 'dart:typed_data';

import '../audio/log_mel_extractor.dart';

class IntermediateFeatureWriter {
  static Future<Directory> createRunDirectory({
    required String rootPath,
    required String wavPath,
  }) async {
    final root = Directory(rootPath);

    if (!await root.exists()) {
      await root.create(recursive: true);
    }

    final timestamp = DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('.', '-');
    final audioName = wavPath
        .split(Platform.pathSeparator)
        .last
        .replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
    final dir = Directory('${root.path}/${timestamp}_$audioName');

    await dir.create(recursive: true);
    return dir;
  }

  static Future<void> writePreprocessing({
    required Directory directory,
    required Float32List resampledWaveform,
    required LogMelExtractionResult extraction,
  }) async {
    await _writeVector(
      '${directory.path}/resampled_waveform.csv',
      resampledWaveform,
    );
    await _writeVector('${directory.path}/padded_waveform.csv',
        extraction.paddedWaveform);
    await _writeMatrix('${directory.path}/frames.csv', extraction.frames);
    await _writeVector('${directory.path}/fft_window.csv', extraction.fftWindow);
    await _writeMatrix(
      '${directory.path}/power_spectrum.csv',
      extraction.powerSpectrum,
    );
    await _writeMatrix('${directory.path}/mel_basis.csv', extraction.melBasis);
    await _writeMatrix('${directory.path}/mel_output.csv', extraction.melOutput);
    await _writeMatrix('${directory.path}/logmel.csv', extraction.logMel);
    await _writeMatrix('${directory.path}/per_bin_mean_std.csv', [
      extraction.perBinMean,
      extraction.perBinStd,
    ]);
    await _writeMatrix(
      '${directory.path}/normalized_features.csv',
      extraction.normalizedFeatureMatrix,
    );
  }

  static Future<void> writeInference({
    required Directory directory,
    required List<double> encoderOutput,
    required List<int> encoderOutputShape,
    required List<double> ctcOutput,
    required List<int> ctcOutputShape,
    required List<int> tokenIds,
  }) async {
    await _writeVector('${directory.path}/encoder_output.csv', encoderOutput);
    await _writeVector(
      '${directory.path}/ctc_logits_or_log_probs.csv',
      ctcOutput,
    );
    await _writeVector('${directory.path}/token_ids.csv', tokenIds);

    await File('${directory.path}/metadata.txt').writeAsString(
      [
        'encoder_output_shape=${encoderOutputShape.join(",")}',
        'ctc_output_shape=${ctcOutputShape.join(",")}',
        'token_count=${tokenIds.length}',
      ].join('\n'),
      flush: true,
    );
  }

  static Future<void> _writeVector(
    String path,
    Iterable<num> values,
  ) async {
    await File(path).writeAsString(
      values.map((value) => value.toString()).join('\n'),
      flush: true,
    );
  }

  static Future<void> _writeMatrix(
    String path,
    Iterable<Iterable<num>> matrix,
  ) async {
    await File(path).writeAsString(
      matrix.map((row) => row.map((value) => value.toString()).join(',')).join(
            '\n',
          ),
      flush: true,
    );
  }
}
