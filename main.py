import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import sys
sys.path.append('/home/dagarwal/Speech-Articulatory-Coding')
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
    all_arrays = []
    for array in arrays:
        additional_padding = (chunk_size - len(array) % chunk_size) % chunk_size
        padded_array = np.pad(array, ((0, additional_padding), (0, 0)), mode='constant', constant_values=0)
        split_arrays = np.array_split(padded_array, len(padded_array) // chunk_size)
        all_arrays += split_arrays
    return all_arrays

def load_and_transform_file(file_path, chunk_size):
    file_prefix = os.path.basename(file_path).split("_")[0]
    file_pitch_stats = pitch_stats_data[file_prefix]
    
    data = np.load(file_path, allow_pickle=True).item()
    ema_data = data['ema']
    pitch_data = (np.log(data['pitch']) - np.log(file_pitch_stats[0]))[:-1].reshape(-1, 1)
    loudness_data = data['loudness'][:-1].reshape(-1, 1)
    combined_data = np.concatenate([ema_data, pitch_data, loudness_data], axis=1)
    return combined_data

def get_features(chunk_size, print_every=2000):
    file_paths = [os.path.join(root, file) 
                  for root, _, files in os.walk(articulatory_feature_directory) 
                  for file in files if file.endswith(".npy")]

    # Process files in parallel
    total_files = len(file_paths)
    processed_files = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers= (int)(cpu_count() / 3)) as executor:
        results = []
        for idx, result in enumerate(executor.map(load_and_transform_file, file_paths, [chunk_size] * total_files)):
            if result is not None:
                results.append(result)
            processed_files += 1
            if processed_files % print_every == 0:
                print(f"Processed {processed_files} / {total_files} files")

    # Concatenate and chunk results
    combined_data = np.concatenate([res for res in results if res is not None], axis=0)
    if chunk_size != 1:
        chunked_data = np.transpose(np.array(create_chunks([combined_data], chunk_size)), (0, 2, 1))
    else:
        chunked_data = combined_data[:, :, np.newaxis]
    return chunked_data

    #         if idx % 200 == 0:
    #             print('here at ', idx)
    #         if idx % 2000 == 0:
    #             with h5py.File(f"/home/dagarwal/processed_articulatory_features.h5", "a") as hf:
    #                 modified_data = np.transpose(np.array(create_chunks(all_data, chunk_size)), (0, 2, 1))
    #                 hf.create_dataset(f"chunks_{idx}", data=modified_data)
    #             with h5py.File("/home/dagarwal/processed_articulatory_features.h5", "r") as hf:
    #                 modified_data = hf[f"chunks_{idx}"][0] 
    #             all_data = []
    #             if idx == 20000:
    #                 break
    # return modified_data

# Call the function

SEQUENCE_LENGTH = 1
TIMESTEPS = 1000

all_data_printing = get_features(SEQUENCE_LENGTH)
print(all_data_printing.shape)
        
# def perform_synthesis(coder, articulatory_feature_file, speaker_embedding):
#     code = np.load(articulatory_feature_file, allow_pickle=True)[()]
#     code['spk_emb'] = speaker_embedding
#     ipd.display(ipd.Audio(wav, rate=coder.sr))
#     return coder.decode(**code)
#     # to display resynthesized audio
    

# speaker_embedding_file = '/home/dagarwal/Speech-Articulatory-Coding/sample_audio/sample1.wav'
# articulatory_feature_file = '/data/common/LibriTTS_R/articulatory_features/100_121669_000001_000000.npy'
# coder = load_model("en", device="cpu")
# speaker_embedding = coder.encode(speaker_embedding_file)['spk_emb']
# perform_synthesis(coder, articulatory_feature_file, speaker_embedding)




model = Unet1D(
    seq_length = SEQUENCE_LENGTH,
    dim = 64,
    # dim_mults = (1, 2, 4, 8),
    channels = 14
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = SEQUENCE_LENGTH,
    timesteps = TIMESTEPS,
    objective = 'pred_x0'
)

training_seq = torch.from_numpy(all_data_printing).float()
# training_seq = torch.rand(64, 14, SEQUENCE_LENGTH) # features are normalized from 0 to 1

# with accelerator.autocast():
#     loss = diffusion(training_seq)
# loss = diffusion(training_seq)
# loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 2048,
    train_lr = 8e-5,
    train_num_steps = 120000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every = 500
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)



noisy_sample = diffusion.q_sample(training_seq[-1].unsqueeze(0).to('cuda:0'), t = torch.tensor([500]).to('cuda:0'))
# denoised_sample = noisy_sample
# for t in reversed(range(diffusion.num_timesteps)):
#     denoised_sample, _ = diffusion.p_sample(denoised_sample, t)
