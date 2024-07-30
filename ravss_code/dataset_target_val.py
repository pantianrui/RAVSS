import math
from typing import Dict, List, Union

import numpy as np
import torch
import torch.utils.data as tdata
import soundfile
import pdb
import random

random_number = [2,3,4,5]

class Vox2_Dataset(tdata.Dataset):
    def __init__(self, mix_num, scp_file, ds_root, dstype, visual_embed_type: str = 'resnet', batch_size: int = 4, max_duration: int = 6, sr: int = 16000):
        self._dataframe=[]
        self.batch_size = batch_size
        self.mix_num = mix_num
        self.target_num = 4

        id2dur = {}
        with open(scp_file, 'r') as f:
            for line in f.readlines():
                mix_name = line.rstrip('\n').replace(',','_').replace('/','_')+'.wav'
                line=line.strip().split(',')

                ###########################################################################################################
                uid=line[1]
                for mix_id in range(mix_num):
                    uid += '#'+line[4*mix_id+2]+'/'+line[4*mix_id+3]
                id2dur[uid]=int(float(line[-1])*16000)
                mixture_path= ds_root + f'audio_mixture_{self.mix_num}mix/'+dstype+'/'+mix_name

                s_path = []
                c_path = []
                for mix_id in range(self.target_num):
                    s_path.append(ds_root + 'audio_clean/'+dstype+'/'+line[4*mix_id+2]+'/'+line[4*mix_id+3]+'.wav')
                    c_path.append(ds_root + 'lip/'+dstype+'/'+line[4*mix_id+2]+'/'+line[4*mix_id+3]+'.npy')

                self._dataframe.append({'uid': uid, 'mixture': mixture_path, 's': s_path, 'c': c_path, 'dur': id2dur[uid]})
                ############################################################################################################

        self.visual_embed_type = visual_embed_type
        self._dataframe = sorted(self._dataframe, key=lambda d: d['dur'], reverse=True)
        self._minibatch = []
        start = 0
        while True:
            end = min(len(self._dataframe), start + self.batch_size)
            self._minibatch.append(self._dataframe[start: end])
            if end == len(self._dataframe):
                break
            start = end
        

        self.len = len(self._minibatch)
        self._sr=sr
        self.max_duration = max_duration
        self.max_duration_in_samples = int(max_duration * sr)
        self.max_duration_in_frames = int(max_duration * 25)
        
        # self.spkmap = {}
        # index = 0
        # for item in self._dataframe:
        #     dstype,spkid1, spkid2 = item['uid'].split('#')
        #     spkid1 = spkid1.split('/')[0]
        #     spkid2 = spkid2.split('/')[0]
        #     if spkid1 not in self.spkmap:
        #         self.spkmap[spkid1] = index
        #         index += 1
        #     if spkid2 not in self.spkmap:
        #         self.spkmap[spkid2] = index
        #         index += 1
           
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        batch_list = self._minibatch[index]
        # min_length = batch_list[-1]['dur']
        # min_length_in_second = min_length / 16000.0
        # min_length_in_frame = math.floor(min_length_in_second * 25)
        mixtures = []
        sources = []
        conditions = []
        spkids = None
        for meta_info in batch_list:
            mixture, _ = soundfile.read(meta_info['mixture'], dtype='float32')
            # soundfile.write('mixture.wav',mixture,16000)
            min_length = mixture.shape[0]
            s_id = {}
            c_id = {}

            ####################################################################
            target_num = len(meta_info['c'])
            for mix_id in range(target_num):
                s,_ = soundfile.read(meta_info['s'][mix_id],dtype='float32')
                # soundfile.write(f'clean_{mix_id}.wav',s,16000)
                s_id[mix_id] = s
                c = np.load(meta_info['c'][mix_id]) #(149,1024,1)
                c_id[mix_id] = c
                min_length = min(min_length,s.shape[0])
            ####################################################################
            min_length_in_second = min_length / 16000.0
            min_length_in_frame = math.floor(min_length_in_second * 25)

            mixture = mixture[:min_length]    
            mixture = np.divide(mixture, np.max(np.abs(mixture)))

            for mix_id in range(target_num):
                s_id[mix_id] = s_id[mix_id][:min_length]
                s_id[mix_id] = np.divide(s_id[mix_id], np.max(np.abs(s_id[mix_id])))
                c_id[mix_id] = c_id[mix_id][:min_length_in_frame]

                if self.visual_embed_type == 'resnet':
                    if c_id[mix_id].shape[0] < min_length_in_frame:
                        c_id[mix_id] = np.pad(c_id[mix_id], ((0, min_length_in_frame - c_id[mix_id].shape[0]), (0, 0)), mode = 'edge')
                else:
                    if c_id[mix_id].shape[0] < min_length_in_frame:
                        c_id[mix_id] = np.pad(c_id[mix_id], ((0, min_length_in_frame - c_id[mix_id].shape[0]), (0, 0), (0, 0)), mode = 'edge')
            
                sources.append(s_id[mix_id][:self.max_duration_in_samples])
                conditions.append(c_id[mix_id][:self.max_duration_in_frames])
            mixtures.append(mixture[:self.max_duration_in_samples])

        mixtures = torch.tensor(np.array(mixtures))
        sources = torch.tensor(np.array(sources))
        conditions = torch.tensor(np.array(conditions))

        return mixtures, sources, conditions, spkids

def dummy_collate_fn(x):
    if len(x) == 1:
        return x[0]
    else:
        return x

if __name__ == '__main__':
    model = Vox2_Dataset()
