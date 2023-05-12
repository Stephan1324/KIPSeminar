import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
        self.img_channels = 1
        self.epochs = epochs
        self.data_directory = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER

    # Funktion zum einlesen der Daten und Train/Test Split
    def split(self, test_size=0.2):
        print('---- Train/Test Spilt: ----')
        self.train_data = tf.keras.utils.image_dataset_from_directory(
            self.data_directory,
            # labels='inferred',
            color_mode='grayscale',
            validation_split=test_size,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.test_data = tf.keras.utils.image_dataset_from_directory(
            self.data_directory,
            # labels='inferred',
            color_mode='grayscale',
            validation_split=test_size,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        self.class_names = self.train_data.class_names

        print('Class names:', self.class_names)
        print('     ..... DONE!')

    # Funktion zum Aufbau des Models
    def model_building(self):
        print('---- Model Building: ----')
        # ---------- Model building ----------
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_data = self.train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.test_data = self.test_data.cache().prefetch(buffer_size=AUTOTUNE)

        # Das entsprechende Modell, welches über model_type ausgewählt wurde wird gewählt
        model = Test_Models(img_height=self.img_height, img_width=self.img_width, img_channels=self.img_channels)
        self.model = model.get_model(self.model_type)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Anpassung für Modell 1
        # self.model.compile(optimizer='adam',
        #                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #                    metrics=['accuracy'])

        # Funktion erstellt ein summary des Modells
        # self.model.summary()
        print('     ..... DONE!')

    # Funktion zum trainieren des Modesll
    def fit(self):
        print('---- Training: ----')
        self.history = self.model.fit(self.train_data, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_data=self.test_data)

        # Acc
        Model.create_train_validation_plot(history=self.history, epochs=self.epochs)
        print('     ..... DONE!')

        # TODO: Abspeichern des Modells unter einem bestimmten Pfad muss eingefügt werden

    def evaluate(self):
        print('---- Evaluation of Test Data: ----')
        # auf den Testdaten werden die Wahrscheinlichkeiten der Labels vorhergesagt
        test_probabilities = self.model.predict(self.test_data)
        # Labels der Vorhersage werden erstellt
        test_predictions = np.argmax(test_probabilities, axis=-1)

        test_labels = np.concatenate([y for x, y in self.test_data], axis=0)
        test_labels = test_labels.astype(int)

        # Konfusionsmatrix wird erstellt
        Model.create_confusion_matrices(test_labels=test_labels, test_predictions=test_predictions)

        # ROC Kurve wird erstellt
        Model.create_roc_curve(test_labels=test_labels, test_probabilities=test_probabilities)
        print('     ..... DONE!')

    # Funktion zu Vorhersage auf einen einzelnes Bild
    # TODO: !!! funktioniert aktuell nicht !!!
    def predict(self, img_path):
        test_image_path = img_path
        test_image = tf.keras.preprocessing.image.load_img(
            test_image_path,
            color_mode='grayscale',
            target_size=(self.img_height, self.img_width),
        )
        test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image_array = tf.expand_dims(test_image_array, 0)

        # Make a prediction on the test image
        predictions = self.model.predict(test_image_array)
        predicted_label = np.argmax(predictions[0])

        # Display the true label and predicted label
        class_names = self.class_names
        true_label = "pitting"  # Replace with the true label of the test image
        print(f"True label: {true_label}")
        print(f"Predicted label: {class_names[predicted_label]}")

        # Display the test image
        plt.imshow(test_image, cmap="gray")
        plt.axis("off")
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
    def create_confusion_matrices(cls, test_labels, test_predictions):
        # erstellen der Konfusionsmatrix
        cm = confusion_matrix(test_labels, test_predictions)

        # Plot der Konfusionsmatrix
        plt.matshow(cm, cmap=plt.cm.RdBu)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    @classmethod
    # Methode zum erstellen der ROC Kurve
    def create_roc_curve(cls, test_labels, test_probabilities):
        # erstellen von positive rate, true positive rate and threshold
        fpr, tpr, threshold = roc_curve(test_labels, test_probabilities[:, 0])

        # berechnen der ROC Kurve über die AUC
        roc_auc = auc(fpr, tpr)

        # Plot der ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
