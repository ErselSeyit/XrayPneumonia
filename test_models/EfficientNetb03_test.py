import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix

def evaluate_tflite_model(model_file, test_dir):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define the test data generator
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(300, 300),  # EfficientNetB3 expects 300x300 images
        batch_size=1,  # Process one image at a time
        class_mode='categorical',
        shuffle=False)

    # Get the total number of test samples
    num_samples = test_generator.samples

    # Create arrays to store predictions and true labels
    predictions = np.zeros((num_samples, 2))
    true_labels = np.zeros((num_samples, 2))

    # Evaluate the model on the test data
    for i in range(num_samples):
        images, labels = test_generator[i]

        # Preprocess the images if necessary
        input_data = images.astype(np.float32)

        # Run inference on the image
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the output from the model
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Store the predictions and true labels
        predictions[i] = output_data[0]
        true_labels[i] = labels[0]

    # Get the true labels
    y_true = np.argmax(true_labels, axis=1)

    # Get the predicted labels
    y_pred = np.argmax(predictions, axis=1)

    # Calculate F1 score and confusion matrix
    f1 = f1_score(y_true, y_pred, average='micro')
    cm = confusion_matrix(y_true, y_pred)

    print(f'F1 score: {f1}')
    print(f'Confusion matrix:\n{cm}')

    return f1, cm

# Evaluate the model
model_file = 'EfficientNetB3_model_finetuned.tflite'
test_dir = "/home/siyu/Downloads/archive/chest_xray/test"
evaluate_tflite_model(model_file, test_dir)
