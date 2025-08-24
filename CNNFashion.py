import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# --- Paths ---
zip_path = r"C:\Users\Chris\PycharmProjects\PythonProject6\concrete_data_week4.zip"
extract_path = r"C:\Users\Chris\PycharmProjects\PythonProject6\concrete_data_week4"

# --- Extract dataset ---
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(extract_path))

# --- Dataset directories ---
train_dir = os.path.join(extract_path, 'train')
validation_dir = os.path.join(extract_path, 'valid')
test_dir = os.path.join(extract_path, 'test')

# --- Image data generators ---
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

class_mode = 'categorical' if len(os.listdir(train_dir)) > 2 else 'binary'

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode=class_mode)

val_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode=class_mode)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=1, class_mode=None, shuffle=False)

# --- Load VGG16 base model ---
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False

# --- Build classifier on top ---
x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

if train_generator.num_classes == 2 and class_mode == 'binary':
    output = Dense(1, activation='sigmoid')(x)
    loss = 'binary_crossentropy'
else:
    output = Dense(train_generator.num_classes, activation='softmax')(x)
    loss = 'categorical_crossentropy'

model = Model(inputs=vgg.input, outputs=output)

# --- Compile and train ---
model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=2,
    steps_per_epoch=10,
    validation_data=val_generator,
    validation_steps=10
)

# --- Predict on test data ---
preds = model.predict(test_generator)

# --- Convert predictions to class labels ---
if loss == 'binary_crossentropy':
    predicted_classes = ['Positive' if p > 0.5 else 'Negative' for p in preds[:5]]
else:
    class_indices = train_generator.class_indices
    index_to_label = {v: k for k, v in class_indices.items()}
    predicted_classes = [index_to_label[np.argmax(p)] for p in preds[:5]]

# --- Print first 5 predictions ---
print("First 5 test predictions:")
for i, label in enumerate(predicted_classes):
    print(f"{label}")