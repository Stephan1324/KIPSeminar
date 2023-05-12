import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from CONFIG_GLOBAL import CONFIG_GLOBAL
from Model.Test_Models import Test_Models


class Model:

    # Initialisierung der Klasse Model
    # model_type gibt an, welches Modell trainiert werden soll -> siehe klasse Test_Models
    def __init__(self, model_type='baseline', batch_size=32, epochs=10):
        self.model_type = model_type
        self.batch_size = batch_size
        self.img_height = 150
        self.img_width = 150
        self.img_channels = 3
        self.epochs = epochs
        self.data_directory = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    # Funktion zum einlesen der Daten und Train/Test Split
    def split(self, test_size=0.2, normalize=False):
        print('---- Train/Test Split: ----')

        folder_path = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER

        class_labels = ['KGT_noDefect_simplified', 'KGT_pitting_simplified']

        def load_classes(class_labels, folder_path):

            # Define the data arrays
            x_total = []
            labels = []

            for class_label in class_labels:
                class_path = os.path.join(folder_path, class_label)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_file)
                        if img_path.endswith('.png'):
                            img = cv2.imread(img_path)
                            # einlesen als Graustufen Bild
                            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # einlesen als HSV Bild:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                            img = cv2.resize(img, (self.img_height, self.img_width))
                            img = img / 255.0
                            x_total.append(np.asarray(img).reshape(self.img_height, self.img_width, self.img_channels))
                            labels.append(class_label)
            return np.array(x_total), np.array(labels)

        x_images, labels = load_classes(class_labels, folder_path)

        # normalisieren der Daten der Bilder wenn normalize=True
        if normalize:
            print('     ..... images normalized')
            x_images = x_images.astype('float32')
            mean = x_images.mean()
            std = x_images.std()
            x_images = (x_images - mean) / std
            x_images = (x_images - x_images.min()) / (x_images.max() - x_images.min())

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(labels)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_images, y_encoded,
                                                                                test_size=test_size, random_state=42,
                                                                                stratify=y_encoded)
        print('     ..... DONE!')

    # Funktion zum Aufbau des Models
    def model_building(self):
        print('---- Model Building: ----')
        # ---------- Model building ----------

        # Das entsprechende Modell, welches über model_type ausgewählt wurde wird gewählt
        model = Test_Models(img_height=self.img_height, img_width=self.img_width, img_channels=self.img_channels)
        self.model = model.get_model(self.model_type)

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Funktion erstellt ein summary des Modells
        # self.model.summary()
        print('     ..... used model is:', self.model_type)
        print('     ..... DONE!')

    # Funktion zum trainieren des Modesll
    def fit(self):
        print('---- Training: ----')
        self.history = self.model.fit(x=self.x_train, y=self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_data=(self.x_test, self.y_test))

        # Acc
        Model.create_train_validation_plot(history=self.history, epochs=self.epochs)
        print('     ..... DONE!')

        # TODO: Abspeichern des Modells unter einem bestimmten Pfad muss eingefügt werden

    def evaluate(self):
        print('---- Evaluation of Test Data: ----')
        # auf den Testdaten werden die Wahrscheinlichkeiten der Labels vorhergesagt
        y_pred = self.model.predict(self.x_test)

        # Konfusionsmatrix wird erstellt
        Model.create_confusion_matrices(y_true=self.y_test, y_pred=y_pred)

        # ROC Kurve wird erstellt
        Model.create_roc_curve(y_true=self.y_test, y_pred=y_pred)
        print('     ..... DONE!')

    # Funktion zu Vorhersage auf einen einzelnes Bild
    def predict(self, img_number):
        import matplotlib.pyplot as plt
        img_num = img_number
        img_test = self.x_test[img_num]
        img_test_label = self.y_test[img_num]
        plt.imshow(img_test.reshape(150, 150, 3))
        pred_prob = self.model.predict(tf.expand_dims(img_test, axis=0))
        print("Predicted=%s" % (pred_prob))
        print("Wahres Label: ", img_test_label)
        plt.show()

    @classmethod
    # Methode zum erstellen des train/validation plots
    def create_train_validation_plot(cls, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    @classmethod
    # Methode zum erstellen der Konfusionsmatrix
    def create_confusion_matrices(cls, y_true, y_pred):
        # erstellen der Konfusionsmatrix
        y_pred_binary = (y_pred >= 0.5).astype(int)
        # berechnen der Konfusion matrix
        cm_matrix = confusion_matrix(y_true, y_pred_binary)

        # Plot Konfusion matrix
        plt.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')

        thresh = cm_matrix.max() / 2.0  # for setting the text color

        for i in range(cm_matrix.shape[0]):
            for j in range(cm_matrix.shape[1]):
                plt.text(j, i, format(cm_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm_matrix[i, j] > thresh else "black")

        plt.show()

    @classmethod
    # Methode zum erstellen der ROC Kurve
    def create_roc_curve(cls, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC Kurve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
