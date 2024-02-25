import torch
from cnn import CNNNetwork
import torchaudio
from urbansounddataset import UrbanSoundDataset
import argparse

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1,10) - > [[0.1, 0.2 .... 0.01]]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-annotations_file", required=True)
    parser.add_argument("-audio_dir", required=True)
    parser.add_argument("-model_path", required=True)
    args = parser.parse_args()

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
    cnn.load_state_dict(state_dict)

    # load urban sound dataset
    # instantiating dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )
    usd = UrbanSoundDataset(
        args.annotations_file,
        args.audio_dir,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        "cpu",
    )

    # get a sample from validation dataset for inference
    input, target = usd[0][0], usd[0][1]  # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f" Predicted : {predicted}, Expected : {expected}")
