# First iteration of a reciever to be used as a library

import ldpc_jossy.py.ldpc as ldpc
import numpy as np
from PIL import Image
from scipy.fft import fft, ifft
import simpleaudio as sa
import sounddevice as sd
from scipy.io.wavfile import write, read
import scipy
from transmitter import LDPC

QFSK_dictionary = {
    (1,1) : -1-1j,
    (1,0) :  1-1j,
    (0,0) :  1+1j,
    (0,1) : -1+1j}


def pilot_decoded_mag_phase(tds, ofdm):
    fds = fft(tds, int(ofdm.N))
    channel_mags = []
    channel_phases = []
    
    for i in range(len(fds)):
        if i in ofdm.pilot_locs:
            channel_mags.append(np.abs(fds[i]))
            channel_phases.append(np.angle(fds[i]))
    
    return np.array(channel_mags), np.array(channel_phases)


def pilot_channel_estimator(tds, ofdm, interval = np.linspace(0, 1023, 1024)):
    dcds, angles = pilot_decoded_mag_phase(tds, ofdm)
    angles = angles % np.pi
    adcd = np.abs(ofdm.pilot_vals)
    aangle = np.angle(ofdm.pilot_vals) % np.pi

    mag_fit = scipy.interpolate.CubicSpline(ofdm.pilot_locs, dcds / adcd)
    ang_fit = scipy.interpolate.CubicSpline(ofdm.pilot_locs, angles - aangle)

    mgf = mag_fit(interval)
    agf = ang_fit(interval)

    H_est = mgf * np.exp(1j*agf)

    return H_est


def pilot_equaliser(tds, ofdm, interval = np.linspace(0, 1023, 1024)):
    H_est = pilot_channel_estimator(tds, ofdm, interval)

    fds = fft(tds, int(ofdm.N))[:int(ofdm.N / 2)]

    return fds / H_est



def deconstruct(aud, ofdm, numpy_func = True, channel_H = None, retSymbs = False):
    N = ofdm.N
    L = ofdm.CP
    QFSK_dict  = ofdm.QFSK_dict
    backwards_dict = {v : k for k, v in QFSK_dict.items()}

    bits_organised = aud.reshape((-1, L+N))

    cut_bits = bits_organised[:,L:]

    if numpy_func == True:
        freq_data = np.fft.fft(cut_bits)
    else:
        freq_data = fft(cut_bits)


    symbols = freq_data[:,1:int(N/2)]

    if channel_H is None:
        pass
    else:
        symbols = symbols / channel_H

    soliddata=[]
    for i in symbols:
        for j in i:
            if j.real > 0 and j.imag > 0:
                soliddata.extend([backwards_dict[1+1j][0],backwards_dict[1+1j][1]])
            elif j.real < 0 and j.imag > 0:
                soliddata.extend([backwards_dict[-1+1j][0],backwards_dict[-1+1j][1]])
            elif j.real > 0 and j.imag < 0:
                soliddata.extend([backwards_dict[1-1j][0],backwards_dict[1-1j][1]])
            else:
                soliddata.extend([backwards_dict[-1-1j][0],backwards_dict[-1-1j][1]])

    # ldpc = LDPC(rate = ofdm.rate, z = ofdm.z)
    # decoded = ldpc.encode(soliddata)
    
    if retSymbs == True:
        return np.array(soliddata), symbols
    else:
        return np.array(soliddata)
    

def bitsToSaveImage(bit_array, image_name, x, y):
    # turn back into a byte array
    abc = [int((''.join(str(k) for k in j)),base=2) for j in bit_array.reshape(-1,8)]
    # Turn array into form PIL likes
    abc = np.array(abc).reshape((x, y, -1 ))  
    abc = Image.fromarray(abc.astype(np.uint8))

    img = Image.fromarray(np.array(abc))

    img.save('{}.tiff'.format(image_name))


def unrep(datastream):
    array_out = []
    for i in range(0,len(datastream),3):
        if datastream[i]+datastream[i+1]+datastream[i+2] >= 2:
            array_out.append(1)
        else:
            array_out.append(0)
    return np.array(array_out)


def bitArrayToText(bytes_as_bits):
    abc = bytes_as_bits.reshape(-1,8).astype(str)
    abc = [("".join(x)) for x in abc]
    abc = [int(x,base=2) for x in abc]
    abc_bytea = bytearray(abc)
    print(abc_bytea.decode("latin-1"))


def record(timeofrec,fs,filename):

    sd.default.samplerate = fs
    sd.default.channels = 1

    myrecording = sd.rec(int(timeofrec * fs))
    sd.wait()  # Wait until recording is finished

    write('{}.wav'.format(filename), fs, myrecording)  # Save as WAV file


def extractor(symbols, ofdm):
    indices = [i for i in range(ofdm.pilot_locs[0], ofdm.pilot_locs[-1]) if i not in ofdm.pilot_locs]
    return symbols[:,indices]

def snc_extractor(symbols):
    return symbols[[i for i in range(0,len(symbols)) if (i ==0 or i== 1) or i % 60 != 0 and (i-1)%60 != 0]]


def standard_deconstructor(aud, ofdm, channel_H = None, retSymbs = False, ldpc_encoded = True):
    N = ofdm.N
    L = ofdm.CP
    QFSK_dict  = ofdm.QFSK_dict
    backwards_dict = {v : k for k, v in QFSK_dict.items()}

    bits_organised = aud.reshape((-1, L+N))

    snc_removed = snc_extractor(bits_organised)

    cut_bits = snc_removed[:,L:]

    freq_data = fft(cut_bits)

    # symbols = freq_data[:,1:int(N/2)]

    # equalisation

    if channel_H is None:
        pass
    else:
        freq_data = freq_data / channel_H

    symbols = extractor(freq_data, ofdm)

    soliddata=[]
    for i in symbols:
        for j in i:
            if j.real > 0 and j.imag > 0:
                soliddata.extend([backwards_dict[1+1j][0],backwards_dict[1+1j][1]])
            elif j.real < 0 and j.imag > 0:
                soliddata.extend([backwards_dict[-1+1j][0],backwards_dict[-1+1j][1]])
            elif j.real > 0 and j.imag < 0:
                soliddata.extend([backwards_dict[1-1j][0],backwards_dict[1-1j][1]])
            else:
                soliddata.extend([backwards_dict[-1-1j][0],backwards_dict[-1-1j][1]])

    if ldpc_encoded ==True:
        coder = LDPC(rate = ofdm.rate, z = ofdm.z)
        decoded = coder.decode(np.array(soliddata))

        decoded = np.int64(decoded).flatten()
    else:
        decoded = np.array(soliddata)
    
    if retSymbs == True:
        return decoded, symbols
    else:
        return decoded
    
def SchmidlCoxDecoder(audio,ofdm):
    dest = np.arange(0, len(audio))
    P1 = np.zeros(len(dest), dtype=complex)
    for i, d in enumerate(dest):
        P1[i] = sum(audio[d+m].conj()*audio[d+m+int(ofdm.N//2)] for m in range(int(ofdm.N//2))) 
    
    R = np.zeros(len(dest))
    for i, d in enumerate(dest):
        R[i] = sum(abs(audio[d+m+int(ofdm.N//2)])**2 for m in range(int(ofdm.N//2)))

    M = abs(P1)**2/R**2

    b_toPeak = np.ones(ofdm.CP) / ofdm.CP
    a = (1,)
    M_filt = scipy.signal.lfilter(b_toPeak, a, M)
    return M_filt