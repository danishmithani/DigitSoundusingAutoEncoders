"""
pipeline has the following tasks for every audio file
1. Load the file
2. padding the signal (only if needed)
3. extracting the log spectrogram from signal   -   Use of librosa for this step
4. Normalize spectrogram
5. save normalized spectrogram

all of the above actions (in the form of class) are wrapped in class preProcessingPipeline
"""

import os
import pickle
import librosa  # install it from interpreter settings or pip install --user librosa command needed.
import numpy as np


class Loader:  # loads an audio file, which also acts as a wrapper for librosa
    def __init__(self, sample_rate, duration, mono):  # if mono is set true, we are loading audio in mono mode else normal as is audio
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0] # librosa.load returns 2 things, Same rate per second and a 2D array [recorded_samlpes_of_amplitude, Num_of_channels_in_audio]
        return signal


class padder:  # applies padding to array and also acts as a wrapper for numpy pad property
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array,
                 num_missing_items):  # assume we have array, [1,2,3]. we want to pad it with 2 values -> [0,0,1,2,3] (left pad)  mode = constant by default pads array with 0
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)  # can mention how many items we want to prepend with items in num_missing_items and we want to start BEFORE array index (left padding)
        return padded_array

    """We will only use right padding for this project"""

    def right_pad(self, array, num_missing_items):  # assume we have array, [1,2,3]. we want to pad it with 2 values -> [1,2,3,0,0] (right pad)  mode = constant by default pads array with 0
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)  # can mention how many items we want to append with items in num_missing_items and we want to start AFTER array index (right padding)
        return padded_array


class log_spectrogram_extractor:
    """extracts log spectrogram (in dB) from time series signal
    Hence first, we need to extract time series from the signal and then the spectrogram."""

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        # extract short-time fourier transform and returns a 2D array [1+FRAME_SIZE/2, num_frames]. if frame_size = 1024, 1st dim = 513.  for ease of math, we drop last bit, hence 512
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class minmax_normalizer:  # take min value and map it to certain value(say 0), and similarly take max value and map it to say 1
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())  # new min=0, max=1
        norm_array = norm_array * (self.max - self.min) + self.min  # range desired by user
        return norm_array

    def denormalize(self, normalizedarray, original_min, original_max):
        array = (normalizedarray - self.min) / (self.max - self.min)  # inverse of second step in normalize method
        array = array * (original_max - original_min) + original_min
        return array


class saver:
    """saves features and min max values for every spectrogram audio feature."""

    def __init__(self, feature_save_dir, min_max_vales_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_vales_save_dir = min_max_vales_save_dir

    def save_feature(self, norm_feature, file_path):
        save_path = self._generate_save_path(file_path)  # step to create a path to save feature
        np.save(save_path,
                norm_feature)  # step to use that path to save feature. As feature is an array, we can simply use np.save
        return save_path

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]  # split will return 2 parts, [head, tail]. Tail is the file name
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_vales_save_dir, "min_max_vales.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(min_max_values,
              save_path):  # NOte: No self here as its is a static method (connected to the class and not dependent on any object)
        with open(save_path, "wb") as f:
            pickle.dump(min_max_values, f)


class preProcessingPipeline:
    """pipeline has the following tasks for every audio file
    1. Load the file
    2. padding the signal (only if needed)
    3. extracting the log spectrogram from signal   -   Use of librosa for this step
    4. Normalize spectrogram
    5. save normalized spectrogram
    6. store original_min and original_max values for every log spectrogram's which is later helpful during reconstruction of signal

    """

    def __init__(self):
        self._num_of_expected_samples = None
        self._loader = None
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_vales = {}


        # everytime we set the loader, we want the expected samples to be calculated at the same time
    @property
    def loader(self):  # this is a method which is now accessed as an attribute which helps get value of another  hidden attributr
        return self._loader

    @loader.setter  # Helps set value of a hidden attribute.
    def loader(self, loader):
        self._loader = loader
        self._num_of_expected_samples = int(loader.sample_rate * loader.duration)

        """We could have added above classes as methods in this class like we did in other files.
         but preprocessing is not only about extracting spectrogram. It may have other functions.
         Hence for this class to have a more generic outlook, we tried this approach"""

    """Here we now loop through every audio file,
        check if padding is needed, and apply if any needed,
        then extract features 
        normalise features,
        save features and then finally save min_max values which we will use later during regeneration of the audio files"""

    def process(self, audio_file_dir):
        for root, _, files in os.walk(audio_file_dir):
            for file in files:
                file_path = os.path.join(root, file)  # full file path
                self.processFile(file_path)
                print(f"Processed file: {file_path}")
        # time to store the min-max values dictionary which we created in the above loop
        self.saver.save_min_max_values(self.min_max_vales)

    def processFile(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_needed(signal):
            signal = self._apply_padding(signal)

        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_needed(self, signal):
        # now if the number of samples in the audio files is shorter than expected, we need to add padding

        if len(signal) < self._num_of_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_of_missing_samples = self._num_of_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_of_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, featuremin, featuremax):
        self.min_max_vales[save_path] = {
            "min": featuremin,
            "max": featuremax
        }


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAM_DIR = "D:/Danish_Study_Material/AutoEnc/Spectrograms/"
    MIN_MAX_VALUES_DIR = "D:/Danish_Study_Material/AutoEnc/minmax_fsdd/"
    FILES_DIR = "D:/Danish_Study_Material/AutoEnc/recordings/"

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = padder()
    log_spectrogram_extractor = log_spectrogram_extractor(FRAME_SIZE, HOP_LENGTH)
    minmax_normalizer = minmax_normalizer(0, 1)
    saver = saver(SPECTROGRAM_DIR, MIN_MAX_VALUES_DIR)

    preProcessingPipeline = preProcessingPipeline()
    preProcessingPipeline.loader = loader
    preProcessingPipeline.padder = padder
    preProcessingPipeline.extractor = log_spectrogram_extractor
    preProcessingPipeline.normalizer = minmax_normalizer
    preProcessingPipeline.saver = saver
    preProcessingPipeline.process(FILES_DIR)
