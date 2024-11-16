import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

def evaluate_tflite_model(model_file, test_dir, target_size):
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
        target_size=target_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    # Get the total number of test samples
    num_samples = test_generator.samples

    # Create arrays to store predictions and true labels
    predictions = np.zeros((num_samples, 2))
    true_labels = np.zeros((num_samples, 2))

    # Evaluate the model on the test data
    for i in range(num_samples):
        images, labels = test_generator[i]
        input_data = images.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions[i] = output_data[0]
        true_labels[i] = labels[0]

    y_true = np.argmax(true_labels, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    f1 = f1_score(y_true, y_pred, average='micro')
    cm = confusion_matrix(y_true, y_pred)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(true_labels.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)

    return f1, cm, fpr, tpr, roc_auc

# File paths and target sizes
models = {
    'VGG16': ('/home/siyu/xraypneumonia/assets/VGG16_model_finetuned.tflite', (224, 224)),
    'ResNet50': ('/home/siyu/xraypneumonia/assets/ResNet50_model_finetuned.tflite', (224, 224)),
    'InceptionV3': ('/home/siyu/xraypneumonia/assets/InceptionV3_model_finetuned.tflite', (299, 299)),
    'EfficientNetB3': ('/home/siyu/xraypneumonia/assets/EfficientnetB3_fine_tuning.tflite', (224, 224))
}

test_dir = "/home/siyu/Downloads/archive/chest_xray/test"

# Evaluate models
f1_scores = {}
confusion_matrices = {}
roc_curves = {}

for model_name, (model_file, target_size) in models.items():
    f1, cm, fpr, tpr, roc_auc = evaluate_tflite_model(model_file, test_dir, target_size)
    f1_scores[model_name] = f1
    confusion_matrices[model_name] = cm
    roc_curves[model_name] = (fpr, tpr, roc_auc)

# Plot the F1 scores
plt.figure(figsize=(10, 5))
plt.bar(f1_scores.keys(), f1_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Scores of Different Models')
plt.ylim(0, 1)
plt.show()

# Plot the confusion matrices
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

for i, (model, cm) in enumerate(confusion_matrices.items()):
    ax = axs[i//2, i%2]
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    for (j, k), val in np.ndenumerate(cm):
        ax.text(k, j, f'{val}', ha='center', va='center')
    ax.set_title(f'Confusion Matrix for {model}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

fig.colorbar(cax, ax=axs, orientation='horizontal', fraction=0.1)
plt.show()

# Plot the ROC curves
plt.figure(figsize=(10, 8))

for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.show()
