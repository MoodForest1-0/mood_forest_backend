# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tf_keras as keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from sklearn.metrics import classification_report

# DATASET_DIR = "dataset"
# MODEL_OUT = "moodforest_model.keras"

# IMG_SIZE = (48, 48)
# BATCH_SIZE = 8
# EPOCHS = 20

# EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad']

# # ─────────────────────────────
# # DATA
# # ─────────────────────────────
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=10,
#     zoom_range=0.1,
#     horizontal_flip=True
# )

# train_data = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     color_mode='grayscale',
#     class_mode='categorical',
#     subset='training',
#     classes=EMOTIONS
# )

# val_data = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     color_mode='grayscale',
#     class_mode='categorical',
#     subset='validation',
#     classes=EMOTIONS
# )

# # ─────────────────────────────
# # MODEL
# # ─────────────────────────────
# from tf_keras import layers, models

# def build_model():
#     inp = layers.Input(shape=(48,48,1))

#     x = layers.Conv2D(32,3,activation='relu',padding='same')(inp)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(128,3,activation='relu',padding='same')(x)

#     x = layers.GlobalAveragePooling2D()(x)

#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dropout(0.3)(x)

#     out = layers.Dense(len(EMOTIONS), activation='softmax')(x)

#     return models.Model(inp, out)

# model = build_model()

# # ─────────────────────────────
# # TRAIN
# # ─────────────────────────────
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=EPOCHS
# )

# model.save(MODEL_OUT)

# # ─────────────────────────────
# # EVALUATION
# # ─────────────────────────────
# val_data.reset()
# preds = model.predict(val_data)
# y_pred = np.argmax(preds, axis=1)
# y_true = val_data.classes[:len(y_pred)]

# print(classification_report(y_true, y_pred, target_names=EMOTIONS))

# # ─────────────────────────────
# # GRAPH
# # ─────────────────────────────
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Accuracy")
# plt.legend(["train","val"])
# plt.savefig("accuracy.png")

# print("Training complete ✔")