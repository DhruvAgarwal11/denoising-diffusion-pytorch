import os
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
import sys
sys.path.append('/home/dagarwal/Speech-Articulatory-Coding')
from sparc import load_model
import IPython.display as ipd

# Paths
sample_file = '/data/common/LibriTTS_R/articulatory_features/986_129388_000060_000005.npy'
checkpoint_path = '/home/dagarwal/results/model-2.pt'
pitch_data_file = '/data/common/LibriTTS_R/pitch_stats.npy'
speaker_embedding_file = '/home/dagarwal/Speech-Articulatory-Coding/sample_audio/sample1.wav'
denoised_sample_path = "/home/dagarwal/results/denoised_sample.npy"

# Load pitch stats
pitch_stats_data = np.load(pitch_data_file, allow_pickle=True).item()

# Function to preprocess a single file
def preprocess_sample(file_path):
    file_prefix = os.path.basename(file_path).split("_")[0]
    file_pitch_stats = pitch_stats_data[file_prefix]

    data = np.load(file_path, allow_pickle=True).item()
    ema_data = data['ema']
    pitch_data = (np.log(data['pitch']) - np.log(file_pitch_stats[0]))[:-1].reshape(-1, 1)
    loudness_data = data['loudness'][:-1].reshape(-1, 1)
    
    if ema_data.shape[0] == pitch_data.shape[0] == loudness_data.shape[0]:
        combined_data = np.concatenate([ema_data, pitch_data, loudness_data], axis=1)
        combined_data = combined_data[:, :, np.newaxis]
        return torch.from_numpy(combined_data).float()
    else:
        raise ValueError("Shape mismatch in data components")

if os.path.exists(denoised_sample_path):
    denoised_sample = np.load(denoised_sample_path)
    denoised_sample = torch.from_numpy(denoised_sample).float()
    print("Loaded saved denoised sample.")
else:
    new_sample = preprocess_sample(sample_file)
    start_index = (new_sample.shape[0] - 128) // 2
    new_sample = new_sample[start_index:start_index + 128]
    new_sample = new_sample.permute(2, 1, 0)
    print("Reshaped sample shape:", new_sample.shape)

    # Initialize model and diffusion (same configuration as training)
    model = Unet1D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=14
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=128,     
        timesteps=1000,     
        objective='pred_x0'
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    timestep = 500  
    noisy_sample = diffusion.q_sample(new_sample, torch.tensor([timestep]))
    with torch.no_grad():
        denoised_sample = diffusion.p_sample_loop(noisy_sample.shape)
    np.save(denoised_sample_path, denoised_sample.cpu().numpy())
    print("Saved denoised sample for future use.")

# Extract `ema`, `pitch`, and `loudness` and concatenate correctly
ema_data = denoised_sample[:, :12, :].squeeze().cpu().numpy().transpose()
pitch_data = denoised_sample[:, 12, :].squeeze().cpu().numpy()
loudness_data = denoised_sample[:, 13, :].squeeze().cpu().numpy()
print(ema_data.shape)
print(pitch_data.shape)
print(loudness_data.shape)

# Concatenate to match SPARC model's expected input [1, 14, 128]
# combined_data = np.concatenate((ema_data, pitch_data[np.newaxis, :], loudness_data[np.newaxis, :]), axis=0)
# combined_data = combined_data[np.newaxis, :, :]  # Add batch dimension
# print("Combined data shape for synthesis:", combined_data.shape)

# Load SPARC model and encode speaker embedding
coder = load_model("en", device="cpu")
speaker_embedding = coder.encode(speaker_embedding_file)['spk_emb']
print(speaker_embedding.shape)

# Synthesis function
def perform_synthesis(coder, combined_data, speaker_embedding):
    code_dict = {
        "ema": ema_data,
        "pitch": pitch_data,
        "loudness": loudness_data,
        "spk_emb": speaker_embedding
    }
    wav = coder.decode(**code_dict)
    return wav

# # Perform synthesis and display audio
synthesized_audio = perform_synthesis(coder, None, speaker_embedding)
ipd.display(ipd.Audio(synthesized_audio, rate=coder.sr))
