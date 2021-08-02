import os
from data_utils import decompress_pickle
import numpy

#data_path = 'clean_signals' 

#first_time = True

#reference = decompress_pickle(data_path + '/chb01/chb01_01.edf.pbz2')
#channel_order = reference['metadata']['channels']
#print("Channels: ", channel_order )

#for d in os.listdir(data_path):
#    if d not in ['chb04', 'chb01', 'chb08', 'chb02']:
#        for f in os.listdir(data_path + '/' + d):
#            if f.endswith('.edf.pbz2'):
#                print("Processing File", f)
#                signal_dict = decompress_pickle(data_path + '/' + d + '/' + f)
#                new_order = signal_dict['metadata']['channels']
#                if len(new_order) != len(channel_order): print("LENGTH ERROR!")
#                for i in range(len(channel_order)):
#                    if new_order[i] != channel_order[i]:
#                        print("Channel order:", channel_order[i], " New order:", new_order[i])

#labels = numpy.load('../clean_signals/chb01/chb01_01.labels.npy')
#print(len(labels[0]))

#fbanks = decompress_pickle('../clean_signals/chb01/chb01_01.fbank.pbz2')
#print(fbanks.shape)

#fbanks = fbanks[:-1][:][:]
#fbanks = fbanks[:,:,1]
#print(fbanks)
#print(fbanks.mean(axis=0))
#fbanks -= fbanks.mean(axis=0)
#print(fbanks)
#print(fbanks.mean(axis = 0))

stats = numpy.load('models/eeg_statistics.npy', allow_pickle=True)
print(len(stats))
print(stats[0].shape)
print(stats[1].shape)
print(stats[2][0].shape)
print(stats[2][1].shape)
print(stats[2][2].shape)
print(stats[2][3].shape)
print(stats[3][0].shape)
print(stats[3][1].shape)
print(stats[3][2].shape)
print(stats[3][3].shape)
