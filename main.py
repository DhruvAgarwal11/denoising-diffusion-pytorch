import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

def create_chunks(arrays, chunk_size):
    all_arrays = []
    for array in arrays:
        additional_padding = (chunk_size - len(array) % chunk_size) % chunk_size
        padded_array = np.pad(array, ((0, additional_padding), (0, 0)), mode='constant', constant_values=0)
        split_arrays = np.array_split(padded_array, len(padded_array) // chunk_size)
        separate_arrays = [chunk for chunk in split_arrays]
        all_arrays += separate_arrays
    return all_arrays

def get_features(chunk_size):
    articulatory_feature_directory = "/data/common/LibriTTS_R/articulatory_features"
    pitch_data_file = '/data/common/LibriTTS_R/pitch_stats.npy'
    
    pitch_stats_data = np.load(pitch_data_file, allow_pickle=True)[()]
    all_data = []
    for root, dirs, files in os.walk(articulatory_feature_directory):
        for idx, file in enumerate(files):
            if file.endswith(".npy"):
                file_prefix = file.split("_")[0]
                
                file_pitch_stats = pitch_stats_data[file_prefix]
                print(f"Processing file: {file_prefix}")
                
                file_path = os.path.join(root, file)
                data = np.load(file_path, allow_pickle=True)[()]
                
                ema_data = data['ema']
                
                pitch_data = (np.log(data['pitch']) - np.log(file_pitch_stats[0]))[:-1]
                
                loudness_data = data['loudness'][:-1]
                
                if ema_data.shape[0] == pitch_data.shape[0] == loudness_data.shape[0]:
                    pitch_data = pitch_data.reshape(-1, 1) 
                    loudness_data = loudness_data.reshape(-1, 1)
                    combined_data = np.concatenate([ema_data, pitch_data, loudness_data], axis=1)
                    all_data.append(combined_data)
            if idx == 10:
                break
    return np.transpose(np.array(create_chunks(all_data, chunk_size)), (0, 2, 1))

# Call the function
all_data_printing = get_features(128)
print(all_data_printing.shape)
        



SEQUENCE_LENGTH = 128
TIMESTEPS = 1000

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 14
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = SEQUENCE_LENGTH,
    timesteps = TIMESTEPS,
    objective = 'pred_v'
)

training_seq = torch.from_numpy(all_data_printing).float()
# training_seq = torch.rand(64, 14, SEQUENCE_LENGTH) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)
