from math import ceil, log2

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def put(inpname, outname, M, chan=0):
    fd, data = wavfile.read(inpname)
    # Nk = data.shape[1]
    I_ = data.shape[0]
    # Q = data.dtype

    Lm = 8 * len(M)
    nu = ceil(log2(Lm) + 1)
    K = pow(2, nu + 1)
    N = ceil(I_ / K)

    S = [data[i : i + K :].T for i in range(0, I_, K)]

    last = S[-1]
    S = S[:-1] + [
        np.array(
            [
                np.pad(last[0], (0, K - len(last[0]))),
                np.pad(last[1], (0, K - len(last[1]))),
            ]
        )
    ]

    sigmas = []
    phases = []
    amps = []
    print("Считаем фазы, амплитуды")
    for p in S:
        spectre = np.fft.fft(p[chan])
        sigma, phi, A = spectre, np.angle(spectre), np.abs(spectre)
        sigmas.append(sigma)
        phases.append(phi)
        amps.append(A)

    print("Считаем разности фаз")
    deltas = [np.zeros(sigmas[0].shape)]
    for i in range(1, N):
        deltas.append(phases[i] - phases[i - 1])

    Mbin = bin(int.from_bytes(M, "big"))[2:]

    new_phases = []
    new_phase_segment = np.zeros(K)
    new_phase_segment[0] = phases[0][0]

    print("Собираем новый сегмент")
    for k in range(1, K // 2):
        if k < len(Mbin) + 1:
            if Mbin[k - 1] == "1":
                new_phase_segment[K // 2 - k] = -np.pi / 2
                new_phase_segment[K // 2 + k] = np.pi / 2
            else:
                new_phase_segment[K // 2 - k] = np.pi / 2
                new_phase_segment[K // 2 + k] = -np.pi / 2
        else:
            new_phase_segment[K // 2 - k] = phases[0][K // 2 - k]
            new_phase_segment[K // 2 + k] = phases[0][K // 2 + k]
    new_phase_segment[K // 2] = phases[0][K // 2]

    print("Собираем новый wav")
    new_phases = [new_phase_segment]
    new_sigma_1 = amps[0] * np.exp(1j * new_phase_segment)
    new_S = [np.ceil(np.real(np.fft.ifft(new_sigma_1)))]
    for i in range(1, N):
        new_phase = new_phases[-1] + deltas[i]
        new_phases.append(new_phase)
        new_sigma = amps[i] * np.exp(1j * new_phase)
        new_S.append(np.ceil(np.real(np.fft.ifft(new_sigma))))

    for i, (ph, phnew) in enumerate(zip(phases[:10], new_phases[:10])):
        plt.plot(ph)
        plt.plot(phnew)
        plt.savefig(f"phases/{i}.png")
        plt.clf()

    new_S = np.concatenate(new_S)
    if chan == 0:
        part2 = np.concatenate([p[1] for p in S])
        new_data = np.array([new_S, part2], dtype=np.int16).T
    else:
        part1 = np.concatenate([p[0] for p in S])
        new_data = np.array([part1, new_S], dtype=np.int16).T

    wavfile.write(outname, fd, new_data)
    return K


def extract(filename, K, chan=0):
    fd, data = wavfile.read(filename)
    # Nk = data.shape[1]
    I_ = data.shape[0]
    # Q = data.dtype

    S = data[:K].T[chan]

    spectre = np.fft.fft(S)
    sigma, phi, A = spectre, np.angle(spectre), np.abs(spectre)

    msg = ""
    for t in range(1, K // 2 + 1):
        if phi[K // 2 - t] < -np.pi / 3:
            msg += "1"
        elif phi[K // 2 - t] > np.pi / 3:
            msg += "0"
        else:
            break
    return int(msg, 2).to_bytes(len(msg) // 8 + 1, "big")


filein = "wavfiles/Pink Floyd - Time.wav"
fileout = "stegged/time.wav"
msg = b"New album, huh?"

K = put(filein, fileout, msg)
print(f"Восстановленное сообщение: {extract(fileout, K)}")
