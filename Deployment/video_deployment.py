import os

import cv2
import numpy as np
from tensorflow import keras
from models.model_manager import ModelManager

from CONFIG_GLOBAL import CONFIG_GLOBAL


class VideoDeployment:
    def __init__(self, model_type, epochs=None, batch_size=None, learning_rate=None):
        self.model_type = model_type
        if epochs is None or batch_size is None or learning_rate is None:
            self.model_specification = '_model.h5'
        else:
            self.model_specification = f'_model_{epochs}_{batch_size}_{learning_rate}.h5'

    def load_model(self):
        # Register custom_loss
        with keras.utils.custom_object_scope({'custom_loss': ModelManager.custom_loss}):
            model = keras.models.load_model(
                os.path.join(CONFIG_GLOBAL.PATH_MODELS_FOLDER, self.model_type,
                             self.model_type + self.model_specification))
        return model
        # model = keras.models.load_model(
        #     os.path.join(CONFIG_GLOBAL.PATH_MODEL_FOLDER, self.model_type, self.model_type + self.model_specification))
        # return model

    def normalize_window(self, window):
        window = window.astype('float32')
        mean = window.mean()
        std = window.std()
        window = (window - mean) / std
        window = (window - window.min()) / (window.max() - window.min())
        return window

    def preprocess_window(self, window, hsv=False, normalize=True):
        # Convert the window to the desired color space
        if hsv:
            window = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
        else:
            window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        # Normalize the window
        # window = window / 255.0
        if normalize:
            window = self.normalize_window(window=window)

        return window

    def predict(self, hsv=False, normalize=True):
        model = self.load_model()

        # Set the position and size of the window (x, y, width, height)
        window_x = 730
        window_y = 500
        window_width = 150
        window_height = 150

        # Define colors for the frames
        frame_color_0 = (0, 0, 255)  # Red frame for prediction 0 (B, G, R)
        frame_color_1 = (0, 255, 0)  # Green frame for prediction 1 (B, G, R)

        # Open the video file
        video = cv2.VideoCapture(CONFIG_GLOBAL.PATH_VIDEO)

        # Loop through each frame of the video
        while video.isOpened():
            # Read the current frame
            ret, frame = video.read()

            if not ret:
                # End of video
                break

            # Extract the specified window from the frame
            window = frame[window_y:window_y + window_height,
                           window_x:window_x + window_width]

            # cv2.imshow('Window', window)
            # cv2.waitKey(0)

            # Normalize the window
            normalized_window = self.preprocess_window(
                window, hsv=hsv, normalize=normalize)

            # Convert the window to match the input shape of the model
            input_window = np.expand_dims(normalized_window, axis=0)

            # Perform inference
            predictions = model.predict(input_window)

            # Get the predicted class and probability
            predicted_class = np.argmax(predictions[0])
            prediction_value = predictions[0][predicted_class]

            # Draw a red frame if the prediction is <= 0.5, green frame if the prediction is >0.5
            frame_color = frame_color_1 if prediction_value <= 0.5 else frame_color_0
            predicition_text = 'No Pitting' if prediction_value <= 0.5 else 'Pitting'
            text_color = frame_color_1 if prediction_value <= 0.5 else frame_color_0
            cv2.rectangle(frame, (window_x, window_y), (window_x + window_width, window_y + window_height), frame_color,
                          thickness=6)

            # Display the prediction value over the window
            text = predicition_text + f' | Prediction: {prediction_value:.6f}'
            cv2.putText(frame, text, (window_x - 150, window_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            # Display the frame with results
            cv2.imshow('Video', frame)

            # Wait for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close the window
        video.release()
        cv2.destroyAllWindows()
