import torch
from pathlib import Path
import itertools
import numpy as np
import time
import random
from random import randrange


###Test Streaming DataLoader with PyTorch####
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filePaths, frameLength, hopSize, loadAll=False):
        super(MyIterableDataset).__init__()
        self.loadAll = loadAll
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.filePage = len(self.filePaths)
        self.filePool = list(range(self.filePage))
        print(self.filePage)
        random.shuffle(self.filePool)

        self.currentFileIndx = 0
        self.CurrentEEG = []
        self.CurrentAudio = []
        self.samplePosistions = []

        self.currentSampleIndex = 0
        self.loadDataToBuffer(self.currentFileIndx, loadAll)

        self.samplePosMap = []
        self.generateSamplePostion()

    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    import random

    def loadDataToBuffer(self, fileIndex, loadAll):
        if loadAll == True:
            for i in range(self.filePage):
                self.CurrentEEG.append(
                    np.load(self.filePaths[self.filePool[self.currentFileIndx]][0]).astype(np.float32))
                self.CurrentAudio.append(
                    np.load(self.filePaths[self.filePool[self.currentFileIndx]][1]).astype(np.float32))
        else:
            self.CurrentEEG = np.load(self.filePaths[self.filePool[self.currentFileIndx]][0]).astype(np.float32)
            self.CurrentAudio = np.load(self.filePaths[self.filePool[self.currentFileIndx]][1]).astype(np.float32)

    def generateSamplePostion(self):
        count = 0
        if self.loadAll == True:
            for i in range(self.filePage):
                totalLength, _ = self.CurrentAudio[i].shape
                startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
                self.samplePosMap.append(startPos)
                noData = (totalLength - self.frameLength) // self.hopSize + 1
                assert len(startPos) == noData
                count += noData
            return count
        else:
            for i in range(self.filePage):
                tempAudio = np.load(self.filePaths[i][1]).astype(np.float32)
                totalLength, _ = tempAudio.shape
                startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
                self.samplePosMap.append(startPos)
                noData = (totalLength - self.frameLength) // self.hopSize + 1
                assert len(startPos) == noData
                count += noData
            return count

    def sample_random_data_number_in_one_batch(self, n, total):
        # Return a randomly chosen list of n nonnegative integers summing to total.
        # n: the number of total files    total: batch size
        return [x - 1 for x in self.constrained_sum_sample_pos(n, total + n)]

    def constrained_sum_sample_pos(self, n, total):
        # Return a randomly chosen list of n positive integers summing to total.Each such list is equally likely to occur."""
        dividers = sorted(random.sample(range(1, total), n - 1))
        return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

    def __iter__(self):

        return self

    def __next__(self):
        if self.loadAll == True:
            if self.currentSampleIndex < len(
                    self.samplePosMap[self.filePool[self.currentFileIndx]]):  # still in the same file
                thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                self.currentSampleIndex += 1
                return self.CurrentEEG[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :], \
                       self.CurrentAudio[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :]
            else:  # move to the next file
                # print("next file")
                #### need to shuffle samples from the last file
                random.shuffle(self.samplePosMap[self.filePool[self.currentFileIndx]])
                self.currentFileIndx += 1
                self.currentSampleIndex = 0
                if self.currentFileIndx < self.filePage:  # still in the same iteration
                    # self.loadDataToBuffer(self.currentFileIndx)
                    thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                    self.currentSampleIndex += 1
                    return self.CurrentEEG[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :], \
                           self.CurrentAudio[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :]
                else:
                    # print("here 2")
                    random.shuffle(self.filePool)
                    self.currentFileIndx = 0
                    # self.loadDataToBuffer(self.currentFileIndx)
                    raise StopIteration
                    print("iteration done, restart")
        else:
            if self.currentSampleIndex < len(
                    self.samplePosMap[self.filePool[self.currentFileIndx]]):  # still in the same file
                thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                self.currentSampleIndex += 1
                return self.CurrentEEG[thisEnd - self.frameLength:thisEnd, :], self.CurrentAudio[
                                                                               thisEnd - self.frameLength:thisEnd, :]
            else:  # move to the next file
                # print("next file")
                #### need to shuffle samples from the last file
                random.shuffle(self.samplePosMap[self.filePool[self.currentFileIndx]])
                self.currentFileIndx += 1
                self.currentSampleIndex = 0
                if self.currentFileIndx < self.filePage:  # still in the same iteration
                    self.loadDataToBuffer(self.currentFileIndx)
                    thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                    self.currentSampleIndex += 1
                    return self.CurrentEEG[thisEnd - self.frameLength:thisEnd, :], self.CurrentAudio[
                                                                                   thisEnd - self.frameLength:thisEnd,
                                                                                   :]
                else:
                    # print("here 2")
                    random.shuffle(self.filePool)
                    self.currentFileIndx = 0
                    self.loadDataToBuffer(self.currentFileIndx)
                    raise StopIteration
                    print("iteration done, restart")

import torchaudio.transforms as T
import torch
class CustomAllLoadDataset(torch.utils.data.Dataset):
    def __init__(self, filePaths, frameLength, hopSize):
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.filePage = len(self.filePaths)
        # load all data to memory as continuous data
        self.EEGData = []
        self.AudioData = []
        self.loadDataToBuffer()

        self.sampleIndxMap = []
        self.noData = self.generateSamplePostion()
        print(self.filePage," files loaded. ", self.noData, " training examples loaded")
    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def loadDataToBuffer(self):
        for i in range(self.filePage):
            self.EEGData.append(np.load(self.filePaths[i][0]).astype(np.float32))
            self.AudioData.append(np.load(self.filePaths[i][1]).astype(np.float32))

    def generateSamplePostion(self):
        count = 0
        for i in range(self.filePage):
            totalLength, _ = self.AudioData[i].shape
            startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
            for pos in startPos:
                self.sampleIndxMap.append((i, pos))
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            assert len(startPos) == noData
            count += noData
        return count

    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        return self.EEGData[self.sampleIndxMap[idx][0]][
               self.sampleIndxMap[idx][1] - self.frameLength:self.sampleIndxMap[idx][1], :], self.AudioData[
                                                                                                 self.sampleIndxMap[
                                                                                                     idx][0]][
                                                                                             self.sampleIndxMap[idx][
                                                                                                 1] - self.frameLength:
                                                                                             self.sampleIndxMap[idx][1],
                                                                                             :]


class CustomAllLoadRawWaveDataset(torch.utils.data.Dataset):
    def __init__(self, filePaths, frameLength, hopSize, resample_rate):
        self.filePaths = filePaths
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.resample_rate = resample_rate
        self.filePage = len(self.filePaths)
        # load all data to memory as continuous data
        self.AudioData = []
        self.loadDataToBuffer()

        self.sampleIndxMap = []
        self.noData = self.generateSamplePostion()
        print("data load finished: ", self.filePage, " files in total ", self.noData, " samples in total")
    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def loadDataToBuffer(self):
        for i in range(self.filePage):
            b = np.load(self.filePaths[i])
            data = torch.from_numpy(b['audio'])
            sample_rate = b['fs']
            resampler = T.Resample(sample_rate, self.resample_rate, dtype=data.dtype)
            resampled_waveform = resampler(data)
            self.AudioData.append(resampled_waveform)

    def generateSamplePostion(self):
        count = 0
        for i in range(self.filePage):
            totalLength = self.AudioData[i].shape[0]
            startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
            for pos in startPos:
                self.sampleIndxMap.append((i, pos))
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            assert len(startPos) == noData
            print("file ", i, "load: ", noData)
            count += noData
        return count

    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        return np.array([[[0], [1], [2]]]), torch.unsqueeze(self.AudioData[self.sampleIndxMap[idx][0]][self.sampleIndxMap[idx][1] - self.frameLength:self.sampleIndxMap[idx][1]],0)