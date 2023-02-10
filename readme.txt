timeline of file creation:
1. main.py : created 1st with the class autoencoder. We first work on encoder part, then the decoder part and finally Build Model part.
2. train.py: Now that the skeleton is ready, we input actual data (MNIST) and size of the neural net along with training of 10,000 dataset(for timesaving purposes). We save the model in the file "model" in the same directory to be fetched later. Running this file saves the model
3. analysis.py: We create this file for loading the saved model and visualization of final regeneration and latent space generation. We also discuss 3 important drawbacks of vanilla autoencoder in the very end of this file. running this file loads the model back. along with visualization.

Above is the vanilla autoencoder part.

Now we look at variational autoencoder.
Link to video: https://www.youtube.com/watch?v=b8AzCgY1gZI&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=9
IMportant to watch WHOLE VIDEO before continuing further.

1. VAutoEncoder.py: First we update bottleneck code by adding the function into the graph. Add new method, _calculate_reconstruction_loss.
_cal_kl_loss
and then finally _combined loss

We now add 3000 audio files of digits from link:https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd?resource=download

we create a pipeline file called "preprocess.py"
