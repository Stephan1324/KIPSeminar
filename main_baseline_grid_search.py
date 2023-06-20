from models.baseline import Baseline

if __name__ == "__main__":

    class_labels = ['KGT_noDefect_simplified', 'KGT_pitting_simplified']

    model_instance = Baseline(epochs=10,
                              batch_size=16, initial_learningrate=0.01,
                              class_labels=class_labels)

    x_images, labels = model_instance.load_classes()

    x_images = model_instance.normalize_data(x_images)

    model_instance.split(x_images, labels)

    model_instance.grid_search(augmentation_factor=3, epochs=[5, 10],
                               batch_size=[32, 16],
                               learning_rates=[0.001, 0.01])
