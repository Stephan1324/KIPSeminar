from Deployment.video_deployment import VideoDeployment

if __name__ == "__main__":

    deployment = VideoDeployment(
        model_type='baseline', epochs=10, batch_size=16, learning_rate=0.01)
    deployment.predict(normalize=True, hsv=True)
