import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:logger/logger.dart';

class ClassifierVGG16 {
  late Interpreter _interpreter;
  late List<String> labels;
  final String labelsFileName = 'assets/labels.txt';
  int? numThreads;
  bool _isModelLoaded = false;
  bool _areLabelsLoaded = false;
  final Logger logger = Logger();

  ClassifierVGG16({this.numThreads}) {
    _loadModel().then((_) {
      _isModelLoaded = true;
      if (_areLabelsLoaded) {
        logger.i('Model and labels loaded successfully');
      }
    }).catchError((e) {
      logger.e('Error loading model: $e');
    });

    _loadLabels().then((_) {
      _areLabelsLoaded = true;
      if (_isModelLoaded) {
        logger.i('Model and labels loaded successfully');
      }
    }).catchError((e) {
      logger.e('Error loading labels: $e');
    });
  }

  Future<void> _loadModel() async {
    try {
      var interpreterOptions = InterpreterOptions();
      if (numThreads != null) {
        interpreterOptions.threads = numThreads!;
      }
      _interpreter = await Interpreter.fromAsset(modelName, options: interpreterOptions);
      logger.i('Interpreter loaded successfully');

      var inputShape = _interpreter.getInputTensor(0).shape;
      var outputShape = _interpreter.getOutputTensor(0).shape;

      logger.i('Input Shape: $inputShape');
      logger.i('Output Shape: $outputShape');

      if (inputShape.length != 4 || inputShape[1] != 224 || inputShape[2] != 224 || inputShape[3] != 3) {
        throw Exception('Unexpected input shape: $inputShape. Expected [1, 224, 224, 3].');
      }

      if (outputShape.length != 2 || outputShape[1] != 2) {
        throw Exception('Unexpected output shape: $outputShape. Expected [1, 2].');
      }

      _isModelLoaded = true;
    } catch (e) {
      logger.e('Error loading model: $e');
      rethrow;
    }
  }

  Future<void> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString(labelsFileName);
      labels = labelData.split('\n');
      if (labels.isNotEmpty && labels.last.isEmpty) {
        labels.removeLast();
      }
      logger.i('Labels loaded successfully');
      if (labels.length != 2) {
        throw Exception('Unexpected number of labels: ${labels.length}. Expected 2.');
      }
      _areLabelsLoaded = true;
    } catch (e) {
      logger.e('Error loading labels: $e');
      rethrow;
    }
  }

  Future<Category> predict(img.Image image) async {
    if (!_isModelLoaded || !_areLabelsLoaded) {
      throw StateError('Model or labels not loaded yet');
    }

    // Preprocess the image
    img.Image resizedImage = img.copyResize(image, width: 224, height: 224);
    Float32List inputBytes = Float32List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    for (int y = 0; y < resizedImage.height; y++) {
      for (int x = 0; x < resizedImage.width; x++) {
        int pixel = resizedImage.getPixel(x, y);
        inputBytes[pixelIndex++] = img.getRed(pixel).toDouble();
        inputBytes[pixelIndex++] = img.getGreen(pixel).toDouble();
        inputBytes[pixelIndex++] = img.getBlue(pixel).toDouble();
      }
    }

    // Reshape to input format specific for model
    final input = inputBytes.reshape([1, 224, 224, 3]);

    final output = Float32List(1 * 2).reshape([1, 2]);

    logger.i('Running inference');
    logger.i('Input Tensor Data: ${input.length > 10 ? input.sublist(0, 10) : input}...');

    try {
      _interpreter.run(input, output);
      logger.i('Inference run successfully');
    } catch (e) {
      logger.e('Error during model inference: $e');
      rethrow;
    }

    logger.i('Output values: ${output[0]}');

    double maxElement = output[0].reduce(
      (double maxElement, double element) => element > maxElement ? element : maxElement,
    );
    String label = labels[output[0].indexOf(maxElement)];

    return Category(label, maxElement);
  }

  String get modelName => 'vgg16_model_finetuned.tflite';
}
