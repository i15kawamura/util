import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import re
import scipy.fftpack as scft
import librosa
import librosa.display

def wavread(filename): #wavファイルの読み込み
    
    #dataはすでに正規化済み
    #samplerate : サンプリング周波数[Hz]
    data, samplerate = sf.read(filename, always_2d = True)

    #sampwidth : サンプルサイズ(バイト)の取得
    f = sf.info(filename)
    subtype = f.subtype  #subtype='PCM_16'など
    pattern=r'([+-]?[0-9]+\.?[0-9]*)'  #正規表現
    sampwidth = int(re.search(pattern,subtype).group(0))

    #量子化ビット数[bit]
    bit = sampwidth * 8
    
    #チャネル数
    ch = f.channels

    return data, samplerate, bit, ch, sampwidth

def wavwrite(filename, data, fs): #wavファイルの書き出し
    sf.write(filename, data, fs)

def Const(s,n,snr): #雑音混入信号の定数計算
    s_amp =np.abs(s)
    S = np.mean(s_amp**2)
    n_amp = np.abs(n)
    N = np.mean(n_amp**2)
    c = np.sqrt(S/(N*10**(snr/10)))
    return c

def stft(x,win,step): #短時間フーリエ変換
    l = len(x)
    N = len(win)
    M = int(np.ceil(float(l - N + step ) / step ))
    new_x = np.zeros(N + ((M - 1) * step),dtype = np.float64)
    new_x[: l]=x

    X = np.zeros([M,N],dtype = np.complex128)
    for m in range(M):
        start = step * m
        X[m,:] = scft.fft(new_x[start : start + N] * win , N)
    return X

def istft(X, win, step): #短時間逆フーリエ変換
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = np.zeros(l, dtype = np.float64)
    for m in range(M):
        start = step * m
        ### 滑らかな接続                                                                         
        x[start : start + N] = x[start : start + N] + scft.ifft(X[m, :]).real * win
    return x

def tight_win(win, step):
    N = len(win) 
    tightWin = np.resize(win, (int(np.ceil(N/step)), int(step)))
    tightWin = tightWin.T
    for i in range(int(step)):
      tightWin[i] = tightWin[i]/np.sqrt(sum(np.abs(tightWin[i])**2))
    tightWin = tightWin.T
    tightWin = np.resize(tightWin, (int(step*np.ceil(N/step)), ))
    tightWin = tightWin[:N]
    return tightWin

def spectrogram_show(S,sr,step,title,save_path):
    ax=plt.figure(figsize=(12,4))
    librosa.display.specshow(librosa.amplitude_to_db(S.T,ref=np.max),y_axis='hz',x_axis='time',sr=sr, hop_length=step)
    plt.title(title)
    plt.colorbar(format='%+2.0fdB')
    #bar.ax.set_ylim(0, np.amax(S))
    plt.tight_layout()
    if(save_path==None):
        plt.show()
    else:
        plt.savefig('%s/%s' %(save_path,title))
        plt.show()

def spatial_cepstrum(Recordfiles, win ,step):
    channel = len(Recordfiles)
    t, w = stft(Recordfiles[0, :], win, step).shape
    
    R=np.empty([channel, t, w], dtype=np.complex128)
    for i in range(channel):
        R[i]=np.abs(stft(Recordfiles[i,:]/np.max(Recordfiles[i,:]), win, step))
    a =np.linalg.norm(R, axis=2)/np.sqrt(w)
    q = np.log1p(a)
    Rq = np.einsum("ij, jk -> ik", q, q.T)/t
    D, E = np.linalg.eig(Rq)
    
    d = np.einsum("ij, jk -> ik", E.T, q)
    return d
    
def calculate_steering_vector(mic_alignments, source_locations, sampling_rete, stft_length, sound_speed=340, is_use_far=False):
    freqs=np.linspace(0, sampling_rate, sftf_length)
    n_channels = np.shape(mic_alignments)[1]
    n_sources = np.shape(source_locations)[1]
    if is_use_far==True:
      norm_source_locations=source_locations/np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
      steering_phase=np.einsum('k, ism, ism->ksm', 2.j*np.pi/sound_speed*freqs, norm_source_locations[...,None],mic_alignments[:,None,:])
      steering_vector=1./np.sqrt(n_channels)*np.exp(steering_phase)
      return steering_vector
    else:
      distance=np.sqrt(np.sum(np.square(source_locations[...,None]-mic_alignments[:,None,:]),axis=0))
      delay=distance/sound_speed
      steering_phase=np.einsum('k, sm->ksm', (-2.j)*np.pi*freqs, delay)
      steering_decay_ratio=1./distance
      steering_vector=steering_decay_ratio[None, ...]*np.exp(steering_phase)
      steering_vector=steering_vector/np.linalg.norm(steering_vector,2,axis=2,keepdims=True)
      return steering_vector

def mvdr_beamforming(complex_spectrum, steering_vector):
    Rconv = np.einsum("wmt, wtn -> wmn", complex_spectrum.transpose(2, 0, 1), np.conj(complex_spectrum.transpose(2, 1, 0)))
    Rconv_inv = np.linalg.pinv(Rconv)
    Rconv_inv_a = np.einsum("wnm, wm -> wn", Rconv_inv, steering_vector)
    a_H_Rconv_inverse_a = np.einsum("wn, wn -> w", np.conj(steering_vector), Rconv_inv_a)
    w_mvdr = Rconv_inv_a/np.maximum(a_H_Rconv_inverse_a, 1.e-180)[:, None]
    s_hat = np.einsum("wm, mwt -> wt", np.conj(w_mvdr), complex_spectrum.transpose(0, 2, 1))
    c_hat = np.einsum("wt, wm -> mwt", s_hat, steering_vector)
    return c_hat