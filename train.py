import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from dataset import UrbanSoundDataset
from network import Network

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001

ANNOTATION_FILE = ""
AUDIO_DIR = ""
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print("########################")

    print("Finished training.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        deivce = "cuda"
    else:
        device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATION_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    train_dataloader = create_data_loader(usd, batch_size=BATCH_SIZE)
    cnn = Network().to(device)
    print(cnn)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Trained feed forward net saved at cnn.pth")
