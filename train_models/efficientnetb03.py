import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Define paths and constants
train_dir = "/home/siyu/Downloads/archive/chest_xray/train"
val_dir = "/home/siyu/Downloads/archive/chest_xray/val"
IMG_SIZE = 300  # EfficientNetB3 expects 300x300 images
BATCH_SIZE = 16
NUM_CLASSES = 2  # Assuming binary classification

# Image augmentation layers
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
    layers.RandomZoom(0.2),
]

# Image augmentation function
def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images

# Preprocessing functions
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def input_preprocess_test(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

print("Loading training data...")
# Load the data
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
).map(input_preprocess_train)

ds_val = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
).map(input_preprocess_test)

print("Training data loaded successfully.")
print("Number of training batches:", len(ds_train))
print("Number of validation batches:", len(ds_val))

# Check a single batch of data to ensure it is loaded correctly
for images, labels in ds_train.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

# Define the base model
base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model
base_model.trainable = False

# Add new layers on top
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Define the new model
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Debug: Print model summary
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()

# Define callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1)
]

# Train the model
initial_epochs = 20
print("Starting initial training...")
try:
    history = model.fit(ds_train, validation_data=ds_val, epochs=initial_epochs, callbacks=callbacks)
    print("Initial training completed.")
except Exception as e:
    print(f"Error during initial training: {e}")

# Unfreeze some layers in the base model for fine-tuning
base_model.trainable = True
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training the model for fine-tuning
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
print("Starting fine-tuning...")
try:
    history_fine = model.fit(ds_train,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=ds_val,
                             callbacks=callbacks)
    print("Fine-tuning completed.")
except Exception as e:
    print(f"Error during fine-tuning: {e}")

# Save the fine-tuned model
model.save('efficientnetb3_finetuned_model.h5')
print("Fine-tuned model saved.")

# Convert the fine-tuned model to TensorFlow Lite format
def get_concrete_function(model):
    run_model = tf.function(lambda x: model(x))
    BATCH_SIZE = 1
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3], model.inputs[0].dtype))
    return concrete_func

try:
    concrete_func = get_concrete_function(model)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('EfficientNetB3_model_finetuned.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted and saved successfully.")
except Exception as e:
    print(f"Error during model conversion: {e}")
