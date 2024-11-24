import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import sys
sys.path.append('/home/pmendoza/Speech-Articulatory-Coding')
from sparc import load_model
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import concurrent.futures
from multiprocessing import cpu_count
from accelerate import Accelerator

articulatory_feature_directory = "/data/common/LibriTTS_R/articulatory_features"
pitch_data_file = '/data/common/LibriTTS_R/pitch_stats.npy'

pitch_stats_data = np.load(pitch_data_file, allow_pickle=True).item()

def create_chunks(arrays, chunk_size):
    all_chunks = []
    
    # Calculate the number of chunks (full chunks)
    num_chunks = len(arrays) // chunk_size
        
    # Split the array into chunks of size (chunk_size, 14)
    chunks = arrays[:num_chunks * chunk_size].reshape(-1, chunk_size, 14)
        
    # Append the chunks to the list
    all_chunks.append(chunks)
    
    # Concatenate all chunks from different arrays
    return np.concatenate(all_chunks, axis=0)

def expand_and_pad(arrays, chunk_size):

    num_points, num_features = arrays.shape
    padded_array = np.zeros((num_points, num_features, chunk_size))
    padded_array[:, :, 0] = arrays

    return padded_array

def load_and_transform_file(file_path, chunk_size, sample_size):
    file_prefix = os.path.basename(file_path).split("_")[0]
    file_pitch_stats = pitch_stats_data[file_prefix]
    
    data = np.load(file_path, allow_pickle=True).item()
    ema_data = data['ema']
    pitch_data = (np.log(data['pitch']) - np.log(file_pitch_stats[0]))[:-1].reshape(-1, 1)
    loudness_data = data['loudness'][:-1].reshape(-1, 1)
    combined_data = np.concatenate([ema_data, pitch_data, loudness_data], axis=1)

    sample_indices = np.random.choice(combined_data.shape[0], min(sample_size, combined_data.shape[0]), replace=False)
    return combined_data[sample_indices]

def get_features(chunk_size, sample_size=64, print_every=2000):
    file_paths = [os.path.join(root, file) 
                  for root, _, files in os.walk(articulatory_feature_directory) 
                  for file in files if file.endswith(".npy")]

    # Process files in parallel
    # Change this to be 2000 to see
    total_files = len(file_paths)
    processed_files = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers= (int)(cpu_count() / 3)) as executor:
        results = []
        for _, result in enumerate(executor.map(load_and_transform_file, file_paths, [chunk_size] * total_files, [sample_size] * total_files)):
            if result is not None:
                results.append(result)
            processed_files += 1
            if processed_files % print_every == 0:
                print(f"Processed {processed_files} / {total_files} files")

    # Concatenate and chunk results
    combined_data = np.concatenate([res for res in results if res is not None], axis=0)
    if chunk_size != 1:
        chunked_data = create_chunks(combined_data, chunk_size)
    else:
        chunked_data = combined_data[:, :, np.newaxis]
        
    return np.transpose(chunked_data, (0, 2, 1))


# Call the function

SEQUENCE_LENGTH = 64
SAMPLE_SIZE = 128
TIMESTEPS = 1000

all_data_printing = get_features(chunk_size=SEQUENCE_LENGTH, sample_size=SAMPLE_SIZE)
print(all_data_printing.shape)

model = Unet1D(
    seq_length = SEQUENCE_LENGTH,
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 14,
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = SEQUENCE_LENGTH,
    timesteps = TIMESTEPS,
    objective = 'pred_x0',
    auto_normalize = False
)

training_seq = torch.from_numpy(all_data_printing).float()

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 2**10,
    train_lr = 8e-5,
    train_num_steps = 120000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 1000
)

trainer.train()

# noisy_sample = diffusion.q_sample(training_seq[-1].unsqueeze(0).to('cuda:0'), t = torch.tensor([500]).to('cuda:0'))