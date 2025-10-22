from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
import os, sys

DATASET_PATH = 'datasets/'
MODEL_PATH = 'image_classifier.h5'

num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='training',
    shuffle=True)  # ensure shuffle

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='validation',
    shuffle=False)

new_old = input('create new model? y/n')
if new_old == 'n':
    print('training old model further')
    model = load_model(MODEL_PATH)
elif new_old == 'y':
    print('making new model, if this was an accident, stop the code before the epochs are over')
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid') if class_mode == "binary"
            else Dense(num_classes, activation='softmax')
    ])
else:
    print('unrecognized input, correct input types: "y" or "n"')
    sys.exit()

loss_function = 'binary_crossentropy' if class_mode == 'binary' else 'categorical_crossentropy'
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

test_loss, test_accuracy = model.evaluate(val_data)
print(f'accuracy: {test_accuracy:.2f}')
model.save('image_classifier.h5')