import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

# Set the base directory for the dataset
base_dir = "dataset/train"

# Set the directory paths for training and validation sets
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")
# Set the number of classes and batch size
num_classes = 4
batch_size = 64

# def lr_schedule(epoch):
    # lr = 1e-3
    # if epoch > 2:
        # lr *= 1e-1
    # elif epoch > 4:
        # lr *= 1e-2
    # elif epoch > 6:
        # lr *= 1e-3
    # elif epoch > 8:
        # lr *= 1e-4
    # print('Learning rate:', lr)
    # return lr


# # Set the learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)


# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    vertical_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Data augnmentation for Test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Set the input shape and build the model
input_shape = (256, 256, 3)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Print model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train the model using data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical')


##############################################################
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
#############################################################	
	
	
# Save the class indices
np.save('Model/class_indices.npy', train_generator.class_indices)

# Define the callbacks for saving the model and training plot
checkpoint = ModelCheckpoint("Model/pecs_recognition_model.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1
						     )

csv_logger = CSVLogger('LOGS/training.log')

# Fit the model with callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, csv_logger, lr_scheduler])
    ########################callbacks=[checkpoint, csv_logger, lr_scheduler])

# Save the final model
# model.save("Model/object_recognition_model_final.h5")

#Print model summary
model.summary()

# Evaluate the model on the validation set and print the confusion matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the class labels for the validation set
class_labels = test_generator.class_indices
labels = dict((v, k) for k, v in class_labels.items())
sorted_labels = [labels[k] for k in sorted(labels)]

# Compute the confusion matrix
cm = confusion_matrix(test_generator.classes, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Save the confusion matrix to a file
with open('LOGS/confusion_matrix.txt', 'w') as f:
    f.write('\t'.join(sorted_labels) + '\n')
    for i, row in enumerate(cm):
        f.write(sorted_labels[i] + '\t' + '\t'.join([str(round(x, 2)) for x in row]) + '\n')

