import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from CONFIG_GLOBAL import CONFIG_GLOBAL
from Model.Test_Models import Test_Models


class Model:

    # Initialisierung der Klasse Model
    # model_type gibt an, welches Modell trainiert werden soll -> siehe klasse Test_Models
    def __init__(self, model_type='baseline', batch_size: int = 32, epochs: int = 10):
        self.model_type = model_type
        self.batch_size = batch_size
        self.img_height = 150
        self.img_width = 150
        self.img_channels = 1
        self.epochs = epochs
        self.data_directory = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    # Funktion zum einlesen der Daten und Train/Test Split
    def split(self, test_size=0.2, normalize=False, hsv=False):
        print('\n---- Train/Test Split: ----')

        folder_path = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER

        class_labels = ['KGT_noDefect', 'KGT_pitting']

        if hsv:
            self.img_channels = 3

        # Laden der Bilddaten und Etiketten
        x_images, labels = Model.load_classes(hsv=hsv, class_labels=class_labels, folder_path=folder_path,
                                              img_height=self.img_height, img_width=self.img_width,
                                              img_channels=self.img_channels)

        # Normalisierung der Bilddaten, wenn normalize=True
        if normalize:
            Model.normalize_data(x_images=x_images)

        # Label-Encoding der Etiketten
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(labels)

        # Aufteilung der Daten in Trainings- und Testdaten
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_images, y_encoded,
                                                                                test_size=test_size, random_state=42,
                                                                                stratify=y_encoded)
        print('     ..... Fertig!')

    # Funktion zum Aufbau des Models
    def model_building(self):
        print('\n---- Model Building: ----')

        # Das entsprechende Modell, das über model_type ausgewählt wurde, wird instanziiert
        model = Test_Models(img_height=self.img_height, img_width=self.img_width, img_channels=self.img_channels)
        self.model = model.get_model(self.model_type)

        # Kompilieren des Modells mit Optimizer, Verlustfunktion und Metriken
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Ausgabe einer Zusammenfassung des Modells
        # self.model.summary()
        print('     ..... used model is:', self.model_type)
        print('     ..... DONE!')

    def grid_search(self, epochs=[10, 20], batch_size=[32, 16]):
        epochs = epochs
        batch_size = batch_size

        # Alle möglichen Kombinationen der Hyperparameter generieren
        hyperparams = [(e, b) for e in epochs for b in batch_size]

        # Listen zum Speichern der Ergebnisse
        accuracies = []
        val_accuracies = []

        # Für jede Hyperparameter-Kombination ein neues Modell trainieren und mit pickle abspeichern
        for i, (e, b) in enumerate(hyperparams):
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Trainiere das Modell
            self.model.fit(self.x_train, self.y_train, epochs=e, batch_size=b,
                           validation_data=(self.x_test, self.y_test))

            model_name = f"_model_{e}_{b}.h5"
            self.model.save(
                os.path.join(CONFIG_GLOBAL.PATH_MODEL_FOLDER, self.model_type, self.model_type + model_name))
            accuracy = self.model.history.history['accuracy']
            val_accuracy = self.model.history.history['val_accuracy']
            accuracies.append(accuracy)
            val_accuracies.append(val_accuracy)

        # Konvertiere Listen in NumPy-Arrays für einfachere Visualisierung
        accuracies = np.array(accuracies)
        val_accuracies = np.array(val_accuracies)

        # Plotte die Ergebnisse der Genauigkeit
        plt.figure(figsize=(10, 6))
        for i, (e, b) in enumerate(hyperparams):
            plt.plot(range(1, e + 1), val_accuracies[i], label=f'Epochs: {e}, Batch Size: {b}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Results of Grid Search for Model: {self.model_type}')
        plt.legend()
        plt.show()

    # Funktion zum trainieren des Modesll
    def fit(self, augmentation=True, augmentation_factor=1):
        print('\n---- Training: ----')

        # Funktion zur Aktivierung der Datenaugmentierung
        if augmentation:
            print('     ..... Augmentation!')
            # Generieren von augmentierten Daten mithilfe von ImageDataGenerator
            augmented_x_train, augmented_y_train = Model.augment_data(self.x_train, self.y_train, augmentation_factor=augmentation_factor)
            # Training des Modells mit augmentierten Daten
            self.history = self.model.fit(x=augmented_x_train, y=augmented_y_train, epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_data=(self.x_test, self.y_test))
        else:
            # Training des Modells mit den vorhandenen Trainingsdaten
            self.history = self.model.fit(x=self.x_train, y=self.y_train, epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_data=(self.x_test, self.y_test))

        # Speichern des trainierten Modells
        self.model.save(os.path.join(CONFIG_GLOBAL.PATH_MODEL_FOLDER, self.model_type, self.model_type + '_model.h5'))

        # Erstellen eines Plots zur Darstellung der Trainings- und Validierungsgenauigkeit über die Epochen
        Model.create_train_validation_plot(history=self.history, epochs=self.epochs)
        print('     ..... DONE!')

    def evaluate(self):
        print('\n---- Evaluation of Test Data: ----')
        # Vorhersage der Label-Wahrscheinlichkeiten für die Testdaten
        y_pred = self.model.predict(self.x_test)

        # Erstellen einer Konfusionsmatrix
        Model.create_confusion_matrices(y_true=self.y_test, y_pred=y_pred)

        # Erstellen der ROC-Kurve
        Model.create_roc_curve(y_true=self.y_test, y_pred=y_pred)
        print('     ..... DONE!')

    # Funktion zur Vorhersage auf einem einzelnen Bild
    def predict_image(self, img_number, hsv=True):
        print('\n---- Prediction on image: ----')
        print('     ..... image no.', img_number)
        img_num = img_number
        img_test = self.x_test[img_num]
        img_test_label = self.y_test[img_num]
        if hsv:
            plt.imshow(img_test.reshape(self.img_height, self.img_width, self.img_channels))
        else:
            plt.imshow(img_test.reshape(150, 150), cmap='gray')
        # Vorhersage der Label-Wahrscheinlichkeit für das Bild
        pred_prob = self.model.predict(tf.expand_dims(img_test, axis=0))
        print("Predicted=%s" % (pred_prob))
        print("True Label: ", img_test_label)
        plt.show()
        print('     ..... DONE!')

    @classmethod
    # Methode zum erstellen des train/validation plots
    def create_train_validation_plot(cls, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

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
        plt.show()

    @classmethod
    # Methode zum Erstellen der Konfusionsmatrix
    def create_confusion_matrices(cls, y_true, y_pred):
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

        plt.show()

    @classmethod
    # Methode zum Erstellen der ROC-Kurve
    def create_roc_curve(cls, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot der ROC-Kurve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (Fläche = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falsch-Positive Rate')
        plt.ylabel('Richtig-Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    @classmethod
    def normalize_data(cls, x_images):
        print('     ..... Bilder werden normalisiert')
        x_images = x_images.astype('float32')
        mean = x_images.mean()
        std = x_images.std()
        x_images = (x_images - mean) / std
        x_images = (x_images - x_images.min()) / (x_images.max() - x_images.min())

        return x_images

    @classmethod
    def load_classes(cls, class_labels, folder_path, img_height, img_width, img_channels, hsv=False):
        # Arrays für die Daten definieren
        x_total = []
        labels = []

        if hsv:
            print('     ..... Einlesen als HSV')
        else:
            print('     ..... Einlesen als Schwarz/Weiß')

        for class_label in class_labels:
            class_path = os.path.join(folder_path, class_label)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if img_path.endswith('.png'):
                        img = cv2.imread(img_path)
                        if hsv:
                            # Einlesen als HSV-Bild
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        else:
                            # Einlesen als Graustufenbild
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (img_height, img_width))
                        img = img / 255.0
                        x_total.append(np.asarray(img).reshape(img_height, img_width, img_channels))
                        labels.append(class_label)
        return np.array(x_total), np.array(labels)

    @classmethod
    def augment_data(cls, x_train, y_train, augmentation_factor=1):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        augmented_data = []
        augmented_labels = []

        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]
            x = x.reshape((1,) + x.shape)  # Das Bild wird in die Form (1, Höhe, Breite, Kanäle) umgeformt

            # Generiere augmentierte Bilder
            augmented_images = datagen.flow(x, batch_size=1)
            for j in range(augmentation_factor):
                augmented_image = augmented_images.next()
                augmented_data.append(augmented_image[0])
                augmented_labels.append(y)

        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)

        return augmented_data, augmented_labels

