# First iteration of functional transmitter to be used as a library


import numpy as np
import simpleaudio as sa
from scipy.signal import chirp
import pyaudio
import sounddevice as sd
from scipy.io.wavfile import write, read
from PIL import Image
from scipy.fft import fft, ifft

# Default QFSK constellation
QFSK_dictionary = {
    (1,1) : -1-1j,
    (1,0) :  1-1j,
    (0,0) :  1+1j,
    (0,1) : -1+1j}


# Takes an array of bits and converts it to a 1d array of symbols
def bit2symbol(x, ofdm):
    bits_organised = x.reshape((-1, ofdm.bps)) # organise the bits into lengths of bps
    return np.array([ofdm.QFSK_dict[tuple(b)] for b in bits_organised]) 

# Takes array of symbols and turns it into blocks of size N.
# Will add zeros on the end of the last block if it does not divide nicly. --- need to work on what to add on the end ---
def cut2Blocks(data, ofdm):
    if np.shape(data)[0]%int(ofdm.spb) == 0:
        return data.reshape(-1,int(ofdm.spb))
    else:
        new_data = np.concatenate((data, np.zeros(ofdm.spb - np.shape(data)[0]%int(ofdm.spb))))
        return new_data.reshape(-1,int(ofdm.spb))


def addpilots(data, ofdm):
    di = 0
    pi = 0
    piloted = np.zeros(ofdm.N // 2 - 1)
    for i in range(ofdm.pilot_locs[0], ofdm.pilot_locs[-1] + 1):
        # offset of 1 needed for original functioning of goodSymbols to be retained.
        if i in ofdm.pilot_locs:
            piloted[i-1] = ofdm.pilot_vals[pi]
            pi += 1
        else:
            piloted[i-1] = data[di]
            di += 1

    return piloted




def addpadding(data, ofdm):
    qfsk = np.array(list(ofdm.QFSK_dict.values()))
    padding_low = np.random.choice(qfsk, size=(int(ofdm.start_bin-1)), replace=True)
    padding_high = np.random.choice(qfsk, size=(int(ofdm.N/2-ofdm.end_bin)), replace=True)

    padding_low = np.vstack(np.array([padding_low for i in range(data.shape[0])]))
    padding_high = np.vstack(np.array([padding_high for i in range(data.shape[0])]))
    
    return np.hstack((padding_low, data, padding_high))


# Takes the blocks of symbols and adds the mirrored conjugate onto the end of the blocks
def goodSymbols(data_symbols, ofdm):
    symbols = np.zeros((np.shape(data_symbols)[0],ofdm.N), dtype=complex)
    for i in range(len(data_symbols)):
        piloted = addpilots(data_symbols[i],ofdm)

        for j in range(1, int(ofdm.N/2)):
            symbols[i][j] = piloted[j-1]
            symbols[i][int(ofdm.N)-j] = piloted[j-1]
    return symbols


# inverse dft, default numpy, can be chanegd to scipy
# For some reason the scipy was leaving tiny(e-18) imaginary parts
def inversedft(freq, numpy_func = True):
    if numpy_func == True:
        if len(np.shape(freq)) == 1:
            return np.real(np.fft.ifft(freq))
        else:
            return [np.real(np.fft.ifft(x)) for x in freq]
    else:
        if len(np.shape(freq)) == 1:
            return np.real(ifft(freq))
        else:
            return [np.real(ifft(x)) for x in freq]

# Adds the cyclic prefix
def addGuard(timeDomain, ofdm):
    full_block = np.zeros((np.shape(timeDomain)[0],ofdm.N+ofdm.CP))
    for j in range(len(timeDomain)):
        full_block[j][0:ofdm.CP] = timeDomain[j][-ofdm.CP:]
        full_block[j][ofdm.CP:ofdm.CP+ofdm.N] = timeDomain[j]
    
    return full_block

def bitsFromTiff(image_name):
    # Open image
    image_file = Image.open(image_name)
    image_file_array = np.array(image_file)
    # make into a 1d binary array
    long_array = image_file_array.reshape(1,-1) # Decimal bytes
    bin_byte_array = ["{0:08b}".format(i) for i in long_array[0]] # Binary bytes
    
    return np.array([bity for bytey in bin_byte_array for bity in bytey]).astype(int)

def bitsFromTxt(file):
    f = open(file, 'rb')
    content = f.read()
    return np.array([int(i) for i in ''.join([(format(byte, '08b')) for byte in content])])


def repetionCode(binary, k=3):
    return np.array([element for element in binary for i in range(k)])



def frameMaker(sync, chan_est, data, data_symb_per_frame, zeros_post_sync=np.empty((0,), dtype=int), zeros_post_sybm=np.empty((0,), dtype=int)):
    initial_data_length = len(data)
    message = []
    for i in range(0, initial_data_length, data_symb_per_frame):
        indiv_frame = np.concatenate((sync.flatten(),zeros_post_sync,chan_est,data[i:i+data_symb_per_frame].flatten(),zeros_post_sybm))
        message.append(indiv_frame)
        #data = np.insert(data,i,np.concatenate((sync.flatten(),zeros_post_sync,chan_est,old_data[i:i+data_symb_per_frame].flatten(),zeros_post_sybm)))
    #data = np.insert(data, range(0, len(data), data_symb_per_frame), np.concatenate((sync,zeros_post_sync,chan_est,data[i:i+data_symb_per_frame].flatten(),zeros_post_sybm)))
    return np.array(message)



def fullTrans(data, ofdm):
    symb = bit2symbol(data, ofdm)
    cut_symb = cut2Blocks(symb, ofdm)
    syb_padded = addpadding(cut_symb, ofdm)
    all_symbs = goodSymbols(syb_padded,ofdm)
    return addGuard(inversedft(all_symbs), ofdm)

def audioMaker(frame, name, fs):
    audio = frame.flatten() / np.max(np.abs(frame.flatten()))
    audio_for_file = audio * (2**15 - 1) / np.max(np.abs(audio))
    audio_for_file = audio_for_file.astype(np.int16)
    write('{}.wav'.format(name), fs, audio_for_file)
    return audio_for_file



def double_chirp(fs=44100):
    chirp_time = 1
    t = np.linspace(0, chirp_time, int(chirp_time * fs), False)
    note = chirp(t, f0=500, f1=15000, t1=chirp_time, method='linear')
    note = note*np.hamming(len(note))
    note = np.concatenate((np.zeros(1500),note,note,np.zeros(1500)))
    return note / np.max(np.abs(note))
    

def rand_snc(ofdm):
    qfsk = np.array(list(ofdm.QFSK_dict.values()))
    return np.random.choice(qfsk, size=(int(ofdm.N/2-1)), replace=True)
    ####### to be continueeed


def s_n_c(rands, ofdm):
    cut_symb = cut2Blocks(symb, ofdm)
    syb_padded = addpadding(cut_symb, ofdm)
    all_symbs = goodSymbols(syb_padded,ofdm)
    preamble_array[::2] = 0
    ####### to be continueeed
