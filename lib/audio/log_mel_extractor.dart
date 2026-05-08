import 'dart:math';
import 'dart:typed_data';

class LogMelExtractor {
  final int sampleRate;
  final int nFft;
  final int winLength;
  final int hopLength;
  final int nMels;

  late final Float64List _hannWindow;
  late final List<List<double>> _melFilterbank;

  LogMelExtractor({
    required this.sampleRate,
    required this.nFft,
    required this.winLength,
    required this.hopLength,
    required this.nMels,
  }) {
    _hannWindow = _buildHannWindow(winLength);
    _melFilterbank = _buildMelFilterbank();
  }

  Float32List extract(Float32List waveform) {
    return extractWithIntermediates(waveform).normalizedFeatures;
  }

  LogMelExtractionResult extractWithIntermediates(Float32List waveform) {
    final centeredWaveform = _padForCenteredStft(waveform);
    final numFrames = 1 + ((centeredWaveform.length - nFft) ~/ hopLength);

    if (numFrames <= 0) {
      throw ArgumentError('Audio is too short');
    }

    final features = List.generate(
      numFrames,
      (_) => List<double>.filled(nMels, 0.0),
    );
    final frames = List.generate(
      numFrames,
      (_) => List<double>.filled(nFft, 0.0),
    );
    final powerSpectra = List.generate(
      numFrames,
      (_) => List<double>.filled(nFft ~/ 2 + 1, 0.0),
    );
    final melEnergies = List.generate(
      numFrames,
      (_) => List<double>.filled(nMels, 0.0),
    );

    for (int frame = 0; frame < numFrames; frame++) {
      final start = frame * hopLength;

      final real = Float64List(nFft);
      final imag = Float64List(nFft);
      final windowOffset = (nFft - winLength) ~/ 2;

      for (int i = 0; i < winLength; i++) {
        real[windowOffset + i] =
            centeredWaveform[start + windowOffset + i] * _hannWindow[i];
      }

      for (int i = 0; i < nFft; i++) {
        frames[frame][i] = real[i];
      }

      _fft(real, imag);

      final powerBins = nFft ~/ 2 + 1;
      final power = Float64List(powerBins);

      for (int k = 0; k < powerBins; k++) {
        power[k] = real[k] * real[k] + imag[k] * imag[k];
        powerSpectra[frame][k] = power[k];
      }

      for (int m = 0; m < nMels; m++) {
        double melEnergy = 0.0;

        for (int k = 0; k < powerBins; k++) {
          melEnergy += power[k] * _melFilterbank[m][k];
        }

        melEnergies[frame][m] = melEnergy;
        features[frame][m] = log(max(melEnergy, 1e-10));
      }
    }

    final logMel = features.map((frame) => List<double>.from(frame)).toList();
    final stats = _utteranceMeanNormalization(features);

    final flat = Float32List(numFrames * nMels);
    int index = 0;

    for (int t = 0; t < numFrames; t++) {
      for (int m = 0; m < nMels; m++) {
        flat[index++] = features[t][m];
      }
    }

    return LogMelExtractionResult(
      paddedWaveform: centeredWaveform,
      frames: frames,
      fftWindow: List<double>.from(_hannWindow),
      powerSpectrum: powerSpectra,
      melBasis: _melFilterbank.map((row) => List<double>.from(row)).toList(),
      melOutput: melEnergies,
      logMel: logMel,
      perBinMean: stats.mean,
      perBinStd: stats.std,
      normalizedFeatures: flat,
      normalizedFeatureMatrix: features,
    );
  }

  Float64List _buildHannWindow(int length) {
    final window = Float64List(length);

    for (int i = 0; i < length; i++) {
      window[i] = 0.5 - 0.5 * cos(2 * pi * i / length);
    }

    return window;
  }

  Float32List _padForCenteredStft(Float32List waveform) {
    final pad = nFft ~/ 2;
    final padded = Float32List(waveform.length + pad * 2);

    for (int i = 0; i < padded.length; i++) {
      var sourceIndex = i - pad;

      while (sourceIndex < 0 || sourceIndex >= waveform.length) {
        if (sourceIndex < 0) {
          sourceIndex = -sourceIndex;
        } else {
          sourceIndex = 2 * waveform.length - sourceIndex - 2;
        }
      }

      padded[i] = waveform[sourceIndex];
    }

    return padded;
  }

  List<List<double>> _buildMelFilterbank() {
    final numSpectrogramBins = nFft ~/ 2 + 1;

    const fMin = 0.0;
    final fMax = sampleRate / 2.0;
    final minMel = _hzToMel(fMin);
    final maxMel = _hzToMel(fMax);

    final melPoints = List<double>.generate(
      nMels + 2,
      (i) => minMel + (maxMel - minMel) * i / (nMels + 1),
    );

    final hzPoints = melPoints.map(_melToHz).toList();
    final fftFrequencies = List<double>.generate(
      numSpectrogramBins,
      (i) => i * sampleRate / nFft,
    );

    final filters = List.generate(
      nMels,
      (_) => List<double>.filled(numSpectrogramBins, 0.0),
    );

    for (int m = 0; m < nMels; m++) {
      final lowerHz = hzPoints[m];
      final centerHz = hzPoints[m + 1];
      final upperHz = hzPoints[m + 2];
      final lowerWidth = centerHz - lowerHz;
      final upperWidth = upperHz - centerHz;

      for (int k = 0; k < numSpectrogramBins; k++) {
        final frequency = fftFrequencies[k];
        final lowerSlope = (frequency - lowerHz) / lowerWidth;
        final upperSlope = (upperHz - frequency) / upperWidth;
        final weight = max(0.0, min(lowerSlope, upperSlope));

        if (weight > 0.0) {
          filters[m][k] = weight;
        }
      }

      final slaneyNorm = 2.0 / (upperHz - lowerHz);

      for (int k = 0; k < numSpectrogramBins; k++) {
        filters[m][k] *= slaneyNorm;
      }
    }

    return filters;
  }

  double _hzToMel(double hz) {
    const fSp = 200.0 / 3.0;
    const minLogHz = 1000.0;
    const minLogMel = minLogHz / fSp;
    final logStep = log(6.4) / 27.0;

    if (hz < minLogHz) {
      return hz / fSp;
    }

    return minLogMel + log(hz / minLogHz) / logStep;
  }

  double _melToHz(double mel) {
    const fSp = 200.0 / 3.0;
    const minLogHz = 1000.0;
    const minLogMel = minLogHz / fSp;
    final logStep = log(6.4) / 27.0;

    if (mel < minLogMel) {
      return mel * fSp;
    }

    return minLogHz * exp(logStep * (mel - minLogMel));
  }

  _FeatureStats _utteranceMeanNormalization(List<List<double>> features) {
    final numFrames = features.length;
    final meanByBin = List<double>.filled(nMels, 0.0);
    final stdByBin = List<double>.filled(nMels, 0.0);

    for (int m = 0; m < nMels; m++) {
      double mean = 0.0;

      for (int t = 0; t < numFrames; t++) {
        mean += features[t][m];
      }

      mean /= numFrames;
      meanByBin[m] = mean;

      double variance = 0.0;

      for (int t = 0; t < numFrames; t++) {
        final diff = features[t][m] - mean;
        variance += diff * diff;
      }

      stdByBin[m] = sqrt(variance / numFrames);

      for (int t = 0; t < numFrames; t++) {
        features[t][m] = features[t][m] - mean;
      }
    }

    return _FeatureStats(meanByBin, stdByBin);
  }

  void _fft(Float64List real, Float64List imag) {
    final n = real.length;

    int j = 0;

    for (int i = 1; i < n; i++) {
      int bit = n >> 1;

      while ((j & bit) != 0) {
        j ^= bit;
        bit >>= 1;
      }

      j ^= bit;

      if (i < j) {
        final tempReal = real[i];
        real[i] = real[j];
        real[j] = tempReal;

        final tempImag = imag[i];
        imag[i] = imag[j];
        imag[j] = tempImag;
      }
    }

    for (int len = 2; len <= n; len <<= 1) {
      final angle = -2 * pi / len;
      final wLenReal = cos(angle);
      final wLenImag = sin(angle);

      for (int i = 0; i < n; i += len) {
        double wReal = 1.0;
        double wImag = 0.0;

        for (int k = 0; k < len ~/ 2; k++) {
          final uReal = real[i + k];
          final uImag = imag[i + k];

          final secondIndex = i + k + len ~/ 2;

          final vReal = real[secondIndex] * wReal - imag[secondIndex] * wImag;

          final vImag = real[secondIndex] * wImag + imag[secondIndex] * wReal;

          real[i + k] = uReal + vReal;
          imag[i + k] = uImag + vImag;

          real[secondIndex] = uReal - vReal;
          imag[secondIndex] = uImag - vImag;

          final nextWReal = wReal * wLenReal - wImag * wLenImag;
          final nextWImag = wReal * wLenImag + wImag * wLenReal;

          wReal = nextWReal;
          wImag = nextWImag;
        }
      }
    }
  }
}

class LogMelExtractionResult {
  final Float32List paddedWaveform;
  final List<List<double>> frames;
  final List<double> fftWindow;
  final List<List<double>> powerSpectrum;
  final List<List<double>> melBasis;
  final List<List<double>> melOutput;
  final List<List<double>> logMel;
  final List<double> perBinMean;
  final List<double> perBinStd;
  final Float32List normalizedFeatures;
  final List<List<double>> normalizedFeatureMatrix;

  const LogMelExtractionResult({
    required this.paddedWaveform,
    required this.frames,
    required this.fftWindow,
    required this.powerSpectrum,
    required this.melBasis,
    required this.melOutput,
    required this.logMel,
    required this.perBinMean,
    required this.perBinStd,
    required this.normalizedFeatures,
    required this.normalizedFeatureMatrix,
  });
}

class _FeatureStats {
  final List<double> mean;
  final List<double> std;

  const _FeatureStats(this.mean, this.std);
}
