import os
import pickle
import numpy as np
import soundfile as sf
from FinalsoundGeneration import soundgenerator
from VAutoEncoder import VAutoEncoder
from varTrain import SPECTROGRAM_DIR

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "D:/Danish_Study_Material/AutoEnc/samples/original/"
SAVE_DIR_GENERATED = "D:/Danish_Study_Material/AutoEnc/samples/generated/"
MIN_MAX_VALUES_PATH = "D:/Danish_Study_Material/AutoEnc/minmax_fsdd/min_max_vales.pkl"


# samples a number of spectrograms out of the trained set we have
def sel_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max = [min_max_values[file_path] for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max)
    return sampled_spectrograms, sampled_min_max


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


def load_fsdd(spectrogram_path):
    # to keep things simple, we will  not split data into train and test
    x_train = []
    file_paths = []
    for root, _, filenames in os.walk(spectrogram_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            spectrogram = np.load(file_path)  # (No_Of_FrequencyBins, No_Of_frames)
            x_train.append(spectrogram)  # now all spectrograms are loaded in x train list (not yest a np array type)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    # as conv layer expects arrays with 3 dims, not 2. Now we need to add 3rd dim. for MNIST, it was 1 as it was grey scale, for spectrograms, it is always 3
    x_train = x_train[
        ..., np.newaxis]  # -> (3000, 256, 64, 1)(3000 No of samples, 256 No. of bins, 64 no. of frames, 1 newaxis). This was we fool a network used to grey scale images.
    return x_train, file_paths


if __name__ == "__main__":
    # initialize sound generator
    vae = VAutoEncoder.load("varmodel")
    sound_generator = soundgenerator(vae, HOP_LENGTH)

    # load spectrograms and minmax Values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAM_DIR)

    # sample from all train set, we now sample few spectrograms with their minmax values
    sampled_specs, min_max_values = sel_spectrograms(specs, file_paths, min_max_values, 5)

    # generate audio for sampled spectrograms (By being passed to autoencoder)
    signals, _ = sound_generator.generate(sampled_specs, min_max_values)

    # convert spectrograms to audio (without being passed to autoencoder)
    original_audios =  soundgenerator.convert_spec_audio(sampled_specs, min_max_values)

    # save both the above audios
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_audios, SAVE_DIR_ORIGINAL)
