from scipy.io.wavfile import write
import os
import random
import deeplake
from utils.train_utils import get_project_root

def load(type):
    if type == 'train':
        print('doing train ...')
        path = 'hub://activeloop/nsynth-train'
        NUM_EXAMPLES = 20000
    elif type == 'val':
        print('doing val ...')
        path = 'hub://activeloop/nsynth-val'
        NUM_EXAMPLES = 2000
    elif type == 'test':
        print('doing test ...')
        path = 'hub://activeloop/nsynth-test'
        NUM_EXAMPLES = 4096
    else:
        print("WRONG!")
        return

    ds = deeplake.load(path)

    # Create directory if not exists
    dir_name = type+'_nsynth'
    nsynth_dir = os.path.join(get_project_root(), 'data', dir_name) # This gets the full path to your directory
    full_path = os.path.join(nsynth_dir, 'wav_files') # This gets the full path to your directory

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Get the size of the dataset
    dataset_size = len(ds.audios)
    # Generate a list of 20000 unique random indexes from 0 to dataset_size
    random_indexes = random.sample(range(dataset_size), NUM_EXAMPLES)

    # Define the sample rate of the audio
    sample_rate = 16000

    # Iterate over the random indexes
    for i, rand_idx in enumerate(random_indexes):
        name = ds.note[rand_idx].numpy()[0]
        print(f"Processing {name}")
        file_path = os.path.join(full_path, f'{name}.wav') # Join the full path with your file name
        print(file_path) # This will print the full path to your file
        # Get the example
        example = ds.audios[rand_idx].numpy()

        # Reshape the example
        reshaped_example = example.reshape(64000, )

        # Save the reshaped example as a .wav file
        write(file_path, sample_rate, reshaped_example)


load('train')
load('val')
