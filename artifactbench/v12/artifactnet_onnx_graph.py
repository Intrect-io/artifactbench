"""FP32 STFT/UNet/median-HPSS/global-mel/CNN + FP64 RMS 집계의 단일 출력 그래프.

기존 ArtifactNet의 v95 E2E exporter를 참조하되 현재 extract_features의
곡 전체·res/H/P 공동 top-dB clamp를 유지한다. 독립 청크 추론은 동등하지 않다.
"""
import torch
import torch.nn.functional as F

N_FFT = 2048
HOP = 512


def median_filter(value, kernel, dim):
    pad = kernel//2
    if dim == 2:
        padded = F.pad(value, (pad, pad), mode='reflect')
        length = value.shape[2]
        windows = torch.stack([padded[:, :, i:i+length] for i in range(kernel)], dim=-1)
    elif dim == 1:
        padded = F.pad(value, (0, 0, pad, pad), mode='reflect')
        length = value.shape[1]
        windows = torch.stack([padded[:, i:i+length, :] for i in range(kernel)], dim=-1)
    else:
        raise ValueError('Median dimension must be frequency or time')
    return windows.sort(dim=-1).values[..., kernel//2]


def hpss(magnitude):
    magnitude = magnitude.squeeze(1)
    harmonic = median_filter(magnitude, 31, 2).square()
    percussive = median_filter(magnitude, 31, 1).square()
    denominator = harmonic+percussive+1e-10
    return ((magnitude*harmonic/denominator).unsqueeze(1),
            (magnitude*percussive/denominator).unsqueeze(1))


def features(residual, harmonic, percussive, mel):
    # 이 max/clamp는 모든 청크와 세 성분을 공유한다. 세 번 호출하면 다른 모델이다.
    count = residual.shape[0]
    all_mel = mel(torch.cat([residual, harmonic, percussive], dim=0))
    mr, mh, mp = all_mel[:count], all_mel[count:2*count], all_mel[2*count:]
    delta = F.pad(mr[..., 1:]-mr[..., :-1], (1, 0))
    delta2 = F.pad(delta[..., 1:]-delta[..., :-1], (1, 0))
    return torch.cat([mr, mh, mp, delta, delta2, mh-mp, delta.abs()], dim=1)


def aggregate(probabilities, chunks):
    # rc2: RMS는 FP32에서 계산한 뒤 probability와 함께 NumPy FP64 집계로 넘긴다.
    rms = chunks.square().mean(dim=1).sqrt().to(torch.float64)
    # 0차원 sum + Python float는 TorchScript ONNX에서 FP32로 승격될 수 있다.
    # 1차원 FP64 tensor끼리 더해 분모와 epsilon의 dtype을 끝까지 보존한다.
    total = rms.sum(dim=0, keepdim=True)
    weights = rms/(total+torch.full_like(total, 1e-9))
    return (probabilities.to(torch.float64)*weights).sum().reshape(1)


class RawSongGraph(torch.nn.Module):
    def __init__(self, unet, cnn, mel):
        super().__init__()
        self.unet, self.cnn, self.mel = unet, cnn, mel
        window = torch.hann_window(N_FFT)
        self.register_buffer('window', window)
        samples = torch.arange(N_FFT, dtype=torch.float64)
        bins = torch.arange(N_FFT//2+1, dtype=torch.float64).unsqueeze(1)
        angle = 2*torch.pi*bins*samples/N_FFT
        self.register_buffer('cosine', (angle.cos()*window.double()).float().unsqueeze(1))
        self.register_buffer('sine', (angle.sin()*window.double()).float().unsqueeze(1))

    def stft_magnitude(self, chunks):
        padded = F.pad(chunks.unsqueeze(1), (N_FFT//2, N_FFT//2), mode='reflect')
        real = F.conv1d(padded, self.cosine, stride=HOP)
        imaginary = F.conv1d(padded, self.sine, stride=HOP)
        return (real.square()+imaginary.square()).sqrt().unsqueeze(1)

    def forward(self, audio_chunks):
        magnitude = self.stft_magnitude(audio_chunks)
        residual = self.unet(magnitude)*magnitude
        harmonic, percussive = hpss(residual)
        logits = self.cnn(features(residual, harmonic, percussive, self.mel))
        return aggregate(torch.sigmoid(logits).reshape(-1), audio_chunks)
