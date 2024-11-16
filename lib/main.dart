import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:logger/logger.dart';
import 'classifier_vgg16.dart' as vgg16;
import 'classifier_efficientnet.dart' as efficientnet;
import 'classifier_inceptionv3.dart' as inceptionv3;
import 'classifier_resnet.dart' as resnet;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

void main() {
  runApp(const MyApp());
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.manual, overlays: []); 
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'X-ray Image Classification',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        scaffoldBackgroundColor: const Color(0xFFF0F4F7), 
        textTheme: const TextTheme(
          headlineSmall: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold, color: Colors.white), 
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          titleTextStyle: TextStyle(color: Colors.white, fontSize: 24.0), 
        ),
      ),
      home: const _MyHomePage(title: 'X-ray Pneumonia'),
    );
  }
}

class _MyHomePage extends StatefulWidget {
  const _MyHomePage({this.title});

  final String? title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<_MyHomePage> {
  late dynamic _classifier;
  final List<String> _models = ['VGG16', 'EfficientNet', 'InceptionV3', 'ResNet50'];
  int _currentModelIndex = 0;
  int numThreads = 1;

  var logger = Logger();

  File? _image;
  final picker = ImagePicker();

  Category? category;

  @override
  void initState() {
    super.initState();
    _classifier = _getClassifier(_models[_currentModelIndex], numThreads);
  }

  dynamic _getClassifier(String model, int numThreads) {
    switch (model) {
      case 'VGG16':
        return vgg16.ClassifierVGG16(numThreads: numThreads);
      case 'EfficientNet':
        return efficientnet.ClassifierEfficientNet(numThreads: numThreads);
      case 'InceptionV3':
        return inceptionv3.ClassifierInceptionV3(numThreads: numThreads);
      case 'ResNet50':
        return resnet.ClassifierResNet(numThreads: numThreads);
      default:
        throw Exception('Invalid model name');
    }
  }

  Future<void> pickAnImage() async {
    try {
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          logger.i('Image picked: ${_image!.path}');
          _predict();
        });
      } else {
        logger.i('No image selected.');
      }
    } catch (e) {
      logger.e('Error picking image: $e');
    }
  }

  void _predict() async {
    if (_image != null) {
      img.Image imageInput = img.decodeImage(_image!.readAsBytesSync())!;
      Category pred = await _classifier.predict(imageInput);

      logger.i('Prediction made: ${pred.label} with confidence ${pred.score}');

      setState(() {
        category = pred;
      });
    }
  }

  void _selectModel(int index) {
    setState(() {
      _currentModelIndex = index;
      _classifier = _getClassifier(_models[_currentModelIndex], numThreads);
      Navigator.pop(context); 
    });
  }

  String getAdvice(String label, double confidence) {
    if (label == 'NORMAL') {
      if (confidence < 0.75) {
        return 'Looking good! But hey, remind the patient to keep up with regular exercise and a healthy diet to keep those lungs in tip-top shape!';
      } else {
        return 'All clear! The patient\'s lungs look great. Keep up the good work!';
      }
    } else if (label == 'PNEUMONIA') {
      return 'Signs of pneumonia detected. It might be a good idea to suggest some possible treatments like antibiotics, rest, and hydration. Follow-up with a healthcare provider for further evaluation.';
    }
    return '';
  }

  Color getLabelColor(String label, double confidence) {
    if (label == 'NORMAL') {
      int greenValue = (200 * confidence).toInt(); // Softer green
      return Color.fromARGB(255, 0, greenValue, 0);
    } else if (label == 'PNEUMONIA') {
      int redValue = (200 * confidence).toInt(); // Softer red
      return Color.fromARGB(255, redValue, 0, 0);
    }
    return Colors.black;
  }

  void _showInfoDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Colors.teal, 
          title: const Text('Information', style: TextStyle(color: Colors.white)), 
          content: const SingleChildScrollView(
            child: ListBody(
              children: <Widget>[
                Text('Accuracy:', style: TextStyle(color: Colors.white)), 
                Text('The ratio of correctly predicted instances to the total instances.', style: TextStyle(color: Colors.white)), 
                SizedBox(height: 10),
                Text('F1 Score:', style: TextStyle(color: Colors.white)), 
                Text('A measure of a model’s accuracy, considering both precision and recall.', style: TextStyle(color: Colors.white)), 
                SizedBox(height: 10),
                Text('Confidence:', style: TextStyle(color: Colors.white)), 
                Text('The probability that the prediction is correct.', style: TextStyle(color: Colors.white)), 
              ],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Got it!', style: TextStyle(color: Colors.white)), 
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  void _showAboutDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Colors.teal, 
          title: const Text('About', style: TextStyle(color: Colors.white)), 
          content: const SingleChildScrollView(
            child: ListBody(
              children: <Widget>[
                Text('X-ray Pneumonia', style: TextStyle(color: Colors.white)), 
                SizedBox(height: 10),
                Text('Version 1.0', style: TextStyle(color: Colors.white)), 
                SizedBox(height: 10),
                Text('Developed by: Ersel Seyit', style: TextStyle(color: Colors.white)), 
                SizedBox(height: 10),
                Text('This app classifies X-ray images to detect signs of pneumonia.', style: TextStyle(color: Colors.white)),
              ],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Close', style: TextStyle(color: Colors.white)), 
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.blue, Colors.green],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: const Icon(Icons.menu, color: Colors.white),
                onPressed: () {
                  Scaffold.of(context).openDrawer();
                },
              ),
              Expanded(
                child: Center(
                  child: Text(widget.title!, style: Theme.of(context).textTheme.headlineSmall), 
                ),
              ),
              IconButton(
                icon: const Icon(Icons.info_outline, color: Colors.white),
                onPressed: _showAboutDialog,
              ),
            ],
          ),
        ),
      ),
      drawer: Drawer(
        child: SafeArea(
          child: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.blue, Colors.green],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: ListView(
              padding: const EdgeInsets.all(10.0), 
              children: [
                DrawerHeader(
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Select Model', style: TextStyle(color: Colors.white, fontSize: 24)),
                      IconButton(
                        icon: const Icon(Icons.info_outline, color: Colors.white),
                        onPressed: _showInfoDialog,
                      ),
                    ],
                  ),
                ),
                ..._models.map((model) {
                  int index = _models.indexOf(model);
                  return ListTile(
                    title: Text(model, style: const TextStyle(color: Colors.white)),
                    tileColor: _currentModelIndex == index ? Colors.blue.withOpacity(0.3) : null,
                    selected: _currentModelIndex == index,
                    onTap: () => _selectModel(index),
                  );
                }),
                const SizedBox(height: 20),
                const Divider(color: Colors.white),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 8.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Model Accuracies and F1 Scores:', style: TextStyle(color: Colors.white, fontSize: 18)),
                      SizedBox(height: 8),
                      Text('EfficientNet:', style: TextStyle(color: Colors.white)),
                      Text('Accuracy: 0.9013, F1 Score: 0.9267', style: TextStyle(color: Colors.white)),
                      SizedBox(height: 8),
                      Text('VGG16:', style: TextStyle(color: Colors.white)),
                      Text('Accuracy: 0.9187, F1 Score: 0.9391', style: TextStyle(color: Colors.white)),
                      SizedBox(height: 8),
                      Text('InceptionV3:', style: TextStyle(color: Colors.white)),
                      Text('Accuracy: 0.8955, F1 Score: 0.9251', style: TextStyle(color: Colors.white)),
                      SizedBox(height: 8),
                      Text('ResNet50:', style: TextStyle(color: Colors.white)),
                      Text('Accuracy: 0.9370, F1 Score: 0.9542', style: TextStyle(color: Colors.white)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      body: Column(
        children: <Widget>[
          SizedBox(
            height: (MediaQuery.of(context).size.height - kToolbarHeight) * 0.5,
            child: _image == null
                ? const Center(
                    child: Text('Please pick an image to continue.', style: TextStyle(fontSize: 16.0, color: Colors.black)), 
                  )
                : Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Image.file(_image!, fit: BoxFit.cover),
                    ),
                  ),
          ),
          const SizedBox(height: 16),
          Center(
            child: Text(
              category != null ? category!.label : 'Result will be shown here',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: category != null ? getLabelColor(category!.label, category!.score) : Colors.black,
              ), 
            ),
          ),
          const SizedBox(height: 8),
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              decoration: BoxDecoration(
                color: Colors.teal.shade50,
                borderRadius: BorderRadius.circular(50.0),
              ),
              child: Text(
                category != null
                    ? getAdvice(category!.label, category!.score)
                    : '',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.black),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ],
      ),
      bottomNavigationBar: SizedBox(
        height: 60.0,
        child: BottomAppBar(
          color: Colors.blue,
          shape: const CircularNotchedRectangle(),
          notchMargin: 6.0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: <Widget>[
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Text(
                  _models[_currentModelIndex],
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.white),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 5.0),
                child: Text(
                  category != null ? 'Confidence: ${category!.score.toStringAsFixed(2)}' : '',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.white),
                ),
              ),
            ],
          ),
        ),
      ),
      floatingActionButton: ShaderMask(
        shaderCallback: (Rect bounds) {
          return const LinearGradient(
            colors: <Color>[Colors.blue, Colors.green],
            begin: Alignment.centerLeft,
            end: Alignment.centerRight,
          ).createShader(bounds);
        },
        child: FloatingActionButton(
          onPressed: pickAnImage,
          backgroundColor: Colors.teal,
          shape: const CircleBorder(),
          child: const Icon(Icons.photo),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }
}
