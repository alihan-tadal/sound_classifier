import torch
import torchaudio

from network import Network
from dataset import UrbanSoundDataset
from scripts.train import AUDIO_DIR, ANNOTATION_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jack_hammer",
    "siren",
    "street_music",
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


if __name__ == "__main__":
    cnn = Network()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )
    usd = UrbanSoundDataset(
        ANNOTATION_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, "cpu"
    )
