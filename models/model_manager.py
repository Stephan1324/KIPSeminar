import os

import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc

from CONFIG_GLOBAL import CONFIG_GLOBAL


class ModelManager:

    # Initialisierung der Klasse Model
    def __init__(self, batch_size: int = 32,
                 epochs: int = 10, initial_learningrate=0.01, hsv=False,
                 img_height=150, img_width=150, img_channels=1,
                 class_labels=None):

        if class_labels is None:
            class_labels = ['KGT_noDefect', 'KGT_pitting']
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_learning_rate = initial_learningrate
        self.img_channels = 1
        self.hsv = hsv
        self.img_height = img_height
        self.img_width = img_width
        if hsv:
            self.img_channels = 3
        else:
            self.img_channels = img_channels

        # Fix default Parameters
        self.lr_scheduler = LearningRateScheduler(self.lr_schedule, verbose=1)
        self.data_directory = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER
        self.class_labels = class_labels

        self.model = None
        self.model_type = None

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    # Funktion zum einlesen der Daten und Train/Test Split
    def split(self, x_images, labels, test_size=0.2):

        print('\n---- Train/Test Split: ----')
        print('     ..... Test Size: ' + str(test_size * 100) + '%')
        # Label-Encoding der Etiketten
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(labels)

        # Aufteilung der Daten in Trainings- und Testdaten
        x_tr, x_te, y_tr, y_te = train_test_split(x_images, y_encoded,
                                                  test_size=test_size,
                                                  random_state=42,
                                                  stratify=y_encoded)
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        self.y_test = y_te

        print('     ..... Done!')

    # Funktion zum Aufbau des Models
    def model_building(self):

        print('\n---- Model Building: ----')

        # Kompilieren des Modells mit Optimizer, Verlustfunktion und Metriken
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.initial_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy'])


        # Ausgabe einer Zusammenfassung des Modells
        # self.model.summary()
        print('     ..... Used Model is:', self.model_type)
        print('     ..... DONE!')


    # Methode zum durchführen einer Grid Search mit unterschiedlichen BatchSizes und Learning Rates
    # Anzahl der Epochen ist fixiert
    def grid_search(self, augmentation_factor=1,epochs=10, batch_sizes=None, learning_rates=None):
        if batch_sizes is None:
            batch_sizes = [32, 16]
        if learning_rates is None:
            learning_rates = [0.001, 0.01]

        print('\n---- Grid Search: ----')
        batch_sizes = batch_sizes
        learning_rates = learning_rates

        # Alle Kombinationen von Hyperparametern erzeugen
        grid = np.zeros((len(batch_sizes), len(learning_rates)))

        for i, batch_size in enumerate(batch_sizes):
            for j, learning_rate in enumerate(learning_rates):
                #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                self.model.compile(optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=self.initial_learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

                self.model.optimizer.learning_rate.assign(learning_rate)

                augmented_x_train, augmented_y_train = self.augment_data(augmentation_factor=augmentation_factor)
                self.model.fit(augmented_x_train, augmented_y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=(self.x_test, self.y_test))



                accuracy = self.model.history.history['val_accuracy'][-1]
                grid[i, j] = accuracy


                print('Trainingsergebnis: ' + str(i) + str(j))
                print('Batch Size:', batch_size)
                print('Learning Rate:', learning_rate)
                print('Validation Accuracy:', accuracy)
                self.evaluate()

            # Create a heatmap of validation accuracies
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, cmap='Blues', interpolation='nearest')

        for i in range(len(batch_sizes)):
            for j in range(len(learning_rates)):
                plt.text(j, i, f'{grid[i, j] * 100:.2f}%', ha='center', va='center', color='black')

        plt.colorbar(label='Validation Accuracy')
        plt.xticks(np.arange(len(learning_rates)), learning_rates)
        plt.yticks(np.arange(len(batch_sizes)), batch_sizes)
        plt.xlabel('Learning Rate')
        plt.ylabel('Batch Size')
        plt.title('Grid Search Results - Validation Accuracy')
        plt.show()

    # Funktion zum Trainieren des Modesll
    def fit(self, augmentation=True, augmentation_factor=1):
        print('\n---- Training: ----')
        print('     ..... Epochs: ' + str(self.epochs))
        print('     ..... Batch Size: ' + str(self.batch_size))
        print('     ..... Initial Learning Rate: ' + str(self.initial_learning_rate))
        print('     ..... Data Augmentation: ' + str(augmentation))

        # Funktion zur Aktivierung der Data Augmentation
        if augmentation:
            # Generieren von augmentierten Daten
            # mithilfe von ImageDataGenerator
            augmented_x_train, augmented_y_train = self.augment_data(
                augmentation_factor=augmentation_factor)
            # Training des Modells mit augmentierten Daten
            self.history = self.model.fit(x=augmented_x_train,
                                          y=augmented_y_train,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_data=(
                                              self.x_test, self.y_test)
                                          )
                                          #callbacks=[self.lr_scheduler])
        else:
            # Training des Modells mit den vorhandenen Trainingsdaten
            self.history = self.model.fit(x=self.x_train, y=self.y_train,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_data=(
                                              self.x_test, self.y_test),
                                          callbacks=[self.lr_scheduler])

        # Speichern des trainierten Modells
        self.model.save(os.path.join(CONFIG_GLOBAL.PATH_MODELS_FOLDER,
                                     self.model_type, self.model_type + '_model.h5'))

        # Erstellen eines Plots zur Darstellung der Trainings- und
        # Validierungsgenauigkeit über die Epochen
        self.create_train_validation_plot()
        print('     ..... Training DONE!')

    def evaluate(self):
        print('\n---- Evaluation of Test Data: ----')
        # Vorhersage der Label-Wahrscheinlichkeiten für die Testdaten
        y_pred = self.model.predict(self.x_test)

        # Erstellen einer Konfusionsmatrix

        self.create_confusion_matrices(y_true=self.y_test, y_pred=y_pred)

        # Erstellen der ROC-Kurve
        self.create_roc_curve(y_true=self.y_test, y_pred=y_pred)
        print('     ..... Evaluation DONE!')

    # Funktion zur Vorhersage auf einem einzelnen Bild
    def predict_image(self, img_number, hsv=True):
        print('\n---- Prediction on image: ----')
        print('     ..... image no.', img_number)
        img_num = img_number
        img_test = self.x_test[img_num]
        img_test_label = self.y_test[img_num]
        if hsv:
            plt.imshow(img_test.reshape(self.img_height,
                                        self.img_width, self.img_channels))
        else:
            plt.imshow(img_test.reshape(150, 150), cmap='gray')
        # Vorhersage der Label-Wahrscheinlichkeit für das Bild
        pred_prob = self.model.predict(tf.expand_dims(img_test, axis=0))
        print('     ..... Predicted=%s' % (pred_prob))
        print('     ..... True Label: ', img_test_label)
        plt.show()
        print('     ..... DONE!')

    # Methode zum erstellen des train/validation plots
    def create_train_validation_plot(self):
        print('     ..... Create Train/Validation Plot')
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        # Erstellen des Plots für die Trainings- und Validierungsgenauigkeit
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        # Erstellen des Plots für den Trainings- und Validierungsverlust
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        print('     ..... DONE!')
        plt.show()

    # Methode zum Erstellen der Konfusionsmatrix
    def create_confusion_matrices(self, y_true, y_pred):
        print('     ..... Create Confusion Matrix!')
        # Erstellen der Konfusionsmatrix
        y_pred_binary = (y_pred >= 0.5).astype(int)
        # Berechnen der Konfusionsmatrix
        cm_matrix = confusion_matrix(y_true, y_pred_binary)

        # Plot der Konfusionsmatrix
        plt.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')

        thresh = cm_matrix.max() / 2.0  # für die Farbe des Textes festlegen

        # Hinzufügen der Werte in die Konfusionsmatrix
        for i in range(cm_matrix.shape[0]):
            for j in range(cm_matrix.shape[1]):
                plt.text(j, i, format(cm_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm_matrix[i, j] > thresh else "black")
        print('     ..... Done!')
        plt.show()

    # Methode zum Erstellen der ROC-Kurve
    def create_roc_curve(self, y_true, y_pred):
        print('     ..... Create ROC Curve!')
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot der ROC-Kurve
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (Fläche = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falsch-Positive Rate')
        plt.ylabel('Richtig-Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        print('     ..... Done!')
        plt.show()

    def normalize_data(self, x_images):
        print('\n---- Normalize Data: ----')
        x_images = x_images.astype('float32')
        mean = x_images.mean()
        std = x_images.std()
        x_images = (x_images - mean) / std
        x_images = (x_images - x_images.min()) / \
                   (x_images.max() - x_images.min())

        print('     ..... Done!')

        return x_images

    def load_classes(self):
        print('\n---- Load Classes: ----')
        print('     ..... Classes: ' + str(self.class_labels))
        print('     ..... As HSV: ' + str(self.hsv))
        # Arrays für die Daten definieren
        x_total = []
        labels = []

        for class_label in self.class_labels:
            class_path = os.path.join(self.data_directory, class_label)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if img_path.endswith('.png'):
                        img = cv2.imread(img_path)
                        if self.hsv:
                            # Einlesen als HSV-Bild
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        else:
                            # Einlesen als Graustufenbild
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(
                            img, (self.img_height, self.img_width))
                        img = img / 255.0
                        x_total.append(np.asarray(img).reshape(
                            self.img_height, self.img_width,
                            self.img_channels))
                        labels.append(class_label)

        print('     ..... Done!')

        return np.array(x_total), np.array(labels)

    def augment_data(self, augmentation_factor=1):
        print('     ..... Augmentation Factor: ' + str(augmentation_factor))
        np.random.seed(42)
        random.seed(42)
        datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.5,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        augmented_data = []
        augmented_labels = []

        for i in range(len(self.x_train)):
            x = self.x_train[i]
            y = self.y_train[i]
            # Das Bild wird in die Form (1, Höhe, Breite, Kanäle) umgeformt
            x = x.reshape((1,) + x.shape)

            # Generiere augmentierte Bilder
            augmented_images = datagen.flow(x, batch_size=1)
            for j in range(augmentation_factor):
                augmented_image = augmented_images.next()
                augmented_data.append(augmented_image[0])
                augmented_labels.append(y)

        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)

        print('     ..... Augmentation Done!')

        return augmented_data, augmented_labels

    # Definiere die Lernratenfunktion
    def lr_schedule(self, epoch, learning_rate):
        if epoch < 40:
            return learning_rate
        elif epoch < 50:
            return 0.00001
        else:
            return learning_rate * tf.math.exp(-0.1)

    def custom_loss(self, y_true, y_pred, lambda_reg=0.001):
        regularization_loss = tf.reduce_sum(tf.square(y_pred))
        binary_crossentropy_loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred)
        total_loss = binary_crossentropy_loss + lambda_reg * regularization_loss
        return total_loss
