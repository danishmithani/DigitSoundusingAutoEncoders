"""
we have already passed in the spectrograms of train dataset to create a plot of points on a latent space.
Now we will use a new point on the same plane and generate an audio (spectrogram first),
with the help of decoder part
This files helps with generation and processing of new spectrogram.
"""
import librosa
from preprocess import minmax_normalizer

class soundgenerator:
    def __init__(self, vae, hopLength):
        # Hoplength helps to convert from spectrogram to signal
        self.vae = vae
        self.hopLength = hopLength
        self.min_max_normalizer = minmax_normalizer(0, 1)   # WIll help us denormalize spectrogram

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
         # Now we convert spect. to signals
        signals = self.convert_spec_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spec_audio(self, spectrograms, min_max_values):
        signals = []    # converted spectrogram signals come here
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape log spectrogram (removing the extra dimension we had to add in varTrain file as a channel)
            log_spectrogram = spectrogram[:, :, 0]  # copying everything from 1st and 2nd dim, but dropping 3rd one

            # denormalize
            denorm_log_spectrogram = self.min_max_normalizer.denormalize(log_spectrogram, min_max_value["min"], min_max_value["max"])

            # convert from log spec to spectrogram using librosa
            spec = librosa.db_to_amplitude(denorm_log_spectrogram)

            # apply Inverse Short Time Fourier Transform (also called Griffin-Lim/ ISTFT)
            signal = librosa.istft(spec, hop_length=self.hopLength)

            # append current signal to signals list
            signals.append(signal)
        return signals


