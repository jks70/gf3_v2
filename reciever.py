# First iteration of a reciever to be used as a library

import numpy as np
from PIL import Image
from scipy.fft import fft, ifft
import simpleaudio as sa
import sounddevice as sd
from scipy.io.wavfile import write, read


QFSK_dictionary = {
    (1,1) : -1-1j,
    (1,0) :  1-1j,
    (0,0) :  1+1j,
    (0,1) : -1+1j}


def decoded_mag_phase(tds, ofdm):
    fds = fft(tds, int(ofdm.N))
    channel_mags = []
    channel_phases = []
    
    for i in range(len(fds)):
        if i in ofdm.pilot_locs:
            channel_mags.append(np.abs(fds[i]))
            channel_phases.append(np.angle(fds[i]))
    
    return np.array(channel_mags), np.array(channel_phases)


def channel_estimator(tds, ofdm, interval = np.linspace(0, 1023, 1024)):
    dcds, angles = decoded_mag_phase(tds, ofdm)
    angles = angles % np.pi
    adcd = np.abs(ofdm.pilot_vals)
    aangle = np.angle(ofdm.pilot_vals) % np.pi

    mag_fit = scipy.interpolate.CubicSpline(ofdm.pilot_locs, dcds / adcd)
    ang_fit = scipy.interpolate.CubicSpline(ofdm.pilot_locs, angles - aangle)

    mgf = mag_fit(interval)
    agf = ang_fit(interval)

    H_est = mgf * np.exp(1j*agf)

    return H_est


def equaliser(tds, ofdm, interval = np.linspace(0, 1023, 1024)):
    H_est = channel_estimator(tds, ofdm, interval)

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