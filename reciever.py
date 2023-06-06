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
import matplotlib.pyplot as plt
from scipy.signal import chirp

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
    l = symbols.shape[0]
    a1 = np.arange(60, l, 62)
    a2 = np.arange(61, l, 62)
    at = np.concatenate((a1,a2))
    sc_indices = [i for i in range(l) if i not in at] 

    return symbols[sc_indices]


def standard_deconstructor(aud, ofdm, channel_H = None, retSymbs = False, ldpc_encoded = True, add_rotate=None):
    N = ofdm.N
    L = ofdm.CP
    QFSK_dict  = ofdm.QFSK_dict
    backwards_dict = {v : k for k, v in QFSK_dict.items()}

    bits_organised = aud.reshape((-1, L+N))

    snc_removed = snc_extractor(bits_organised)

    cut_bits = snc_removed[:,L:]

    freq_data = fft(cut_bits)

    # symbols = freq_data[:,1:int(N/2)]

    # equalisation

    if channel_H is None:
        pass
    else:
        freq_data = freq_data / channel_H

    symbols = extractor(freq_data, ofdm)
    
    if add_rotate is not None:
        symbols = symbols * np.exp(add_rotate)

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


def channelEstimate(four_known_symbols, ofdm):

    even_one = four_known_symbols[:ofdm.N+ofdm.CP]
    odd_one = four_known_symbols[ofdm.N+ofdm.CP:2*(ofdm.N+ofdm.CP)]
    even_two = four_known_symbols[2*(ofdm.N+ofdm.CP):3*(ofdm.N+ofdm.CP)]
    odd_two = four_known_symbols[3*(ofdm.N+ofdm.CP):4*(ofdm.N+ofdm.CP)]
    odd_index = np.arange(0,741,2)
    even_index=np.arange(1,741,2)

    exported_coding = np.loadtxt('preamble_qpsk_symbols.csv', delimiter=',',dtype=complex)
    exported_coding *= 1.41421356474619

    hes_e1 = (fft(even_one[ofdm.CP:], 2048)).flatten()[49:790] / exported_coding.flatten()
    hes_o1 = (fft(odd_one[ofdm.CP:], 2048)).flatten()[49:790] / exported_coding.flatten()
    hes_e2 = (fft(even_two[ofdm.CP:], 2048)).flatten()[49:790] / exported_coding.flatten()
    hes_o2 = (fft(odd_two[ofdm.CP:], 2048)).flatten()[49:790] / exported_coding.flatten()
    
    hest_one = np.zeros(741, dtype='complex')
    hest_one[even_index] = hes_e1[even_index]
    hest_one[odd_index] = hes_o1[odd_index]

    hest_two = np.zeros(741, dtype='complex')
    hest_two[even_index] = hes_e2[even_index]
    hest_two[odd_index] = hes_o2[odd_index]


    return hest_one, hest_two


def chanest_padd(hest,ofdm):
    full_size = np.ones(ofdm.N, dtype='complex')
    full_size[49:790] = hest
    return full_size

def error_(data1,data_correct):
    errs = 0
    for i in range(len(data_correct)):
        if data_correct[i] != data1[i]:
            errs += 1
    return 100*errs/len(data_correct)

def errorss(decod, bit_array):
    code_length = len(bit_array)
    error_by_block = []
    n=720*2
    for i in range(0,code_length,n):
        error_by_block.append(error_(decod[i:i+n],bit_array[i:i+n]))
    return error_by_block


def chirpEnds(signal, note=None, graph_display=False):

    if note is None:
        chirp_time = 1
        fs =44100
        t = np.linspace(0, chirp_time, int(chirp_time * fs), False)
        note = chirp(t, f0=250, f1=20000, t1=chirp_time, method='linear')
        note = note*np.hamming(len(note))

    norm_signal =  signal/np.max(signal)
    peka = scipy.signal.find_peaks(norm_signal, 0.5)[0]
    start_search1 = peka[0]-10000
    end_search1 = start_search1 + 100000
    end_search2 = peka[-1]+25000
    start_search2 = end_search2 - 100000

    delay_guess1 = np.abs(np.correlate(signal[start_search1:end_search1], note, mode='full'))
    delay_guess2 = np.abs(np.correlate(signal[start_search2:end_search2], note, mode='full'))

    delay_guess_norm1 = delay_guess1/np.max(delay_guess1)
    delay_guess_norm2 = delay_guess2/np.max(delay_guess2)

    peaks_1 = scipy.signal.find_peaks(delay_guess_norm1, 0.5, distance = 3000)
    peaks_2 = scipy.signal.find_peaks(delay_guess_norm2, 0.5, distance = 3000)

    end_first_chirp = peaks_1[0][0]+start_search1
    end_second_chirp = peaks_1[0][1]+start_search1
    end_third_chirp = peaks_2[0][0]+start_search2
    end_fourth_chirp = peaks_2[0][1]+start_search2
    
    if graph_display == True:
        plt.figure(figsize = (30, 10))
        plt.plot(signal)
        plt.axvline(x = start_search1 , color = 'grey',linestyle = '-', label='start_search')
        plt.axvline(x = end_search1 , color = 'grey',linestyle = '--', label='end_search')
        plt.axvline(x = start_search2 , color = 'grey',linestyle = '-', label='start_search')
        plt.axvline(x = end_search2 , color = 'grey',linestyle = '--', label='end_search')

        plt.axvline(x = end_first_chirp , color = 'r', label='end_first_chirp')
        plt.axvline(x = end_second_chirp , color = 'b', label='end_second_chirp')
        plt.axvline(x = end_third_chirp , color = 'r', label='end_third_chirp')
        plt.axvline(x = end_fourth_chirp , color = 'b', label='end_fourth_chirp')

        plt.legend()
        plt.show()


    return [end_first_chirp,end_second_chirp, end_third_chirp, end_fourth_chirp]

def decode_header(bits_in):
    lst = [str(i) for i in bits_in[:7*8*3]]
    out = [int(''.join(map(str, lst[i:i+8])),2) for i in range(0, len(lst), 8)]
    bytes_in = bytearray(out)

    verified_header=b''
    for i in range(7):
        if bytes_in[i] == bytes_in[i+7] or bytes_in[i] == bytes_in[i+14]:
            verified_header += bytes_in[i].to_bytes(1, byteorder='big')
        elif bytes_in[i+7] == bytes_in[i+14]:
            verified_header += bytes_in[i+7].to_bytes(1, byteorder='big')
        else:
            raise ValueError("There are too many errors in the input header")
    filetype = '.' + verified_header[0:4].strip(b'\x00').decode('utf-8')
    filesize = int.from_bytes(verified_header[4:7], 'big')
    return filetype, filesize