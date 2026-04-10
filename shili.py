import soundfile as sf
import soxr
import torch
import yaml
from hydra.utils import instantiate

from addse.lightning import ADDSELightningModule

torch.set_grad_enabled(False)

addse_cfg = "F:/ksl/addse/configs/addse-s.yaml"
addse_ckpt = "F:/ksl/addse/logs/addse-edbase-quick/checkpoints/addse-s.ckpt"
nac_cfg = "F:/ksl/addse/configs/nac.yaml"
nac_ckpt = "F:/ksl/addse/logs/nac/checkpoints/last.ckpt"
audio_path = "F:/ksl/TIMIT_all_wavs/SA1.WAV"
output_path = "F:/ksl/addse/enhanced_SA1.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
with open(addse_cfg) as f:
    cfg = yaml.safe_load(f)
lm: ADDSELightningModule = instantiate(cfg["lm"], nac_cfg=nac_cfg, nac_ckpt=nac_ckpt).to(device)
ckpt = torch.load(addse_ckpt, map_location=device)
lm.load_state_dict(ckpt["state_dict"], strict=False)
lm.eval()

# Load input audio
x, fs = sf.read(audio_path, dtype="float32", always_2d=True)
assert x.shape[1] == 1, "Only mono audio is supported"
x = soxr.resample(x, fs, 16000)
x = torch.from_numpy(x.T).unsqueeze(0).to(device)

# RMS-normalize for best results
rms = x.pow(2).mean().sqrt()
x = x / (rms + 1e-8)

# Enhance audio
with torch.no_grad():
    x_enh = lm(x).squeeze(0)

# Rescale to original RMS
x_enh = x_enh * rms

# Save output audio
sf.write(output_path, x_enh.detach().cpu().T.numpy(), 16000)
print(f"Enhanced audio saved to: {output_path}")