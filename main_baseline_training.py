from models.baseline import Baseline


if __name__ == "__main__":

    class_labels = ['KGT_noDefect_simplified', 'KGT_pitting_simplified']

    model_instance = Baseline(epochs=10,
                              batch_size=16, initial_learningrate=0.01,
                              class_labels=class_labels, hsv=True)

    x_images, labels = model_instance.load_classes()

    x_images = model_instance.normalize_data(x_images)

    model_instance.split(x_images, labels)

    model_instance.model_building()

    model_instance.fit(augmentation=True, augmentation_factor=4)

    model_instance.evaluate()

    model_instance.predict_image(img_number=60, hsv=True)
