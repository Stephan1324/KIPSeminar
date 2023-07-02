from Deployment.video_deployment import VideoDeployment

if __name__ == "__main__":

    deployment = VideoDeployment(
        model_type='BaselineImproved', epochs=None, batch_size=None, learning_rate=None)
    deployment.predict(normalize=True, hsv=True)
