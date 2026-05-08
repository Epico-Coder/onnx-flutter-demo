import 'package:flutter/material.dart';
import 'package:record/record.dart';

import '../services/espnet_asr_service.dart';
import '../services/recording_service.dart';

class RecorderPage extends StatefulWidget {
  const RecorderPage({super.key});

  @override
  State<RecorderPage> createState() => _RecorderPageState();
}

class _RecorderPageState extends State<RecorderPage> {
  final RecordingService _recordingService = RecordingService();
  final EspnetAsrService _asrService = EspnetAsrService();

  bool _isRecording = false;
  bool _isTranscribing = false;
  bool _isLoadingMicrophones = false;

  List<InputDevice> _microphones = [];
  InputDevice? _selectedMicrophone;
  DecodingMode _selectedDecodingMode = DecodingMode.ctcPrefixBeam;
  String? _recordedFilePath;
  String _transcript = '';
  String? _error;

  @override
  void initState() {
    super.initState();
    _initAsr();
    _loadMicrophones();
  }

  Future<void> _initAsr() async {
    try {
      await _asrService.init();
    } catch (e) {
      setState(() {
        _error = 'Failed to initialize ASR: $e';
      });
    }
  }

  Future<void> _loadMicrophones() async {
    setState(() {
      _isLoadingMicrophones = true;
      _error = null;
    });

    try {
      final microphones = await _recordingService.listInputDevices();

      setState(() {
        _microphones = microphones;
        _selectedMicrophone = microphones.contains(_selectedMicrophone)
            ? _selectedMicrophone
            : microphones.isEmpty
            ? null
            : microphones.first;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load microphones: $e';
      });
    } finally {
      setState(() {
        _isLoadingMicrophones = false;
      });
    }
  }

  Future<void> _toggleRecording() async {
    try {
      if (_isRecording) {
        final path = await _recordingService.stop();

        setState(() {
          _isRecording = false;
          _recordedFilePath = path;
        });
      } else {
        final path = await _recordingService.start(device: _selectedMicrophone);

        setState(() {
          _isRecording = true;
          _recordedFilePath = path;
          _transcript = '';
          _error = null;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Recording error: $e';
        _isRecording = false;
      });
    }
  }

  Future<void> _transcribeRecording() async {
    final path = _recordedFilePath;
    if (path == null) return;

    setState(() {
      _isTranscribing = true;
      _transcript = '';
      _error = null;
    });

    try {
      final text = await _asrService.transcribeWav(
        path,
        decodingMode: _selectedDecodingMode,
      );

      setState(() {
        _transcript = text;
      });
    } catch (e) {
      setState(() {
        _error = 'Transcription failed: $e';
      });
    } finally {
      setState(() {
        _isTranscribing = false;
      });
    }
  }

  @override
  void dispose() {
    _recordingService.dispose();
    _asrService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final recordButtonText = _isRecording ? 'Stop Record' : 'Record';
    final canSelectMicrophone = !_isRecording && !_isTranscribing;
    final canSelectDecodingMode = !_isRecording && !_isTranscribing;

    return Scaffold(
      appBar: AppBar(title: const Text('ESPnet ONNX Test')),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              DropdownButtonFormField<InputDevice>(
                initialValue: _selectedMicrophone,
                decoration: InputDecoration(
                  labelText: 'Microphone',
                  suffixIcon: IconButton(
                    onPressed: canSelectMicrophone && !_isLoadingMicrophones
                        ? _loadMicrophones
                        : null,
                    tooltip: 'Refresh microphones',
                    icon: _isLoadingMicrophones
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.refresh),
                  ),
                ),
                items: _microphones
                    .map(
                      (device) => DropdownMenuItem(
                        value: device,
                        child: Text(
                          device.label.isEmpty ? device.id : device.label,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    )
                    .toList(),
                onChanged: canSelectMicrophone
                    ? (device) {
                        setState(() {
                          _selectedMicrophone = device;
                        });
                      }
                    : null,
              ),

              const SizedBox(height: 20),

              DropdownButtonFormField<DecodingMode>(
                initialValue: _selectedDecodingMode,
                decoration: const InputDecoration(
                  labelText: 'Decoding',
                ),
                items: DecodingMode.values
                    .map(
                      (mode) => DropdownMenuItem(
                        value: mode,
                        child: Text(mode.label),
                      ),
                    )
                    .toList(),
                onChanged: canSelectDecodingMode
                    ? (mode) {
                        if (mode == null) return;

                        setState(() {
                          _selectedDecodingMode = mode;
                        });
                      }
                    : null,
              ),

              const SizedBox(height: 20),

              ElevatedButton.icon(
                onPressed: _toggleRecording,
                icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                label: Text(recordButtonText),
              ),

              const SizedBox(height: 20),

              if (_isRecording)
                const Text('Recording...', style: TextStyle(fontSize: 18)),

              if (_recordedFilePath != null && !_isRecording) ...[
                const Text('Saved recording:'),
                const SizedBox(height: 8),
                SelectableText(_recordedFilePath!, textAlign: TextAlign.center),
                const SizedBox(height: 20),
                ElevatedButton.icon(
                  onPressed: _isTranscribing ? null : _transcribeRecording,
                  icon: const Icon(Icons.auto_awesome),
                  label: Text(
                    _isTranscribing ? 'Transcribing...' : 'Run ONNX Test',
                  ),
                ),
              ],

              const SizedBox(height: 24),

              if (_transcript.isNotEmpty) ...[
                const Text(
                  'Transcript:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                SelectableText(_transcript),
              ],

              if (_error != null) ...[
                const SizedBox(height: 24),
                Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
