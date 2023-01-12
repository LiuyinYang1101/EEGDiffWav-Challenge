import torch
from pathlib import Path
import itertools
import numpy as np
import time
import random
from random import randrange
import math
import torchaudio.transforms as T

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


class CustomProcessInOrderDataset(torch.utils.data.Dataset):
    def __init__(self, filePaths, frameLength, hopSize, device):
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.device = device
        self.filePage = len(self.filePaths)
        self.filesIndex = list(range(self.filePage))
        self.halfFile = int(self.filePage*0.5)
        # load all data to memory as continuous data
        self.EEGData = []
        self.AudioData = []
        self.EEG_on_GPU = []
        self.Aud_on_GPU = []
        self.loadDataToBuffer()
        self.convertToTensorType()
        self.noData = self.getNoData()
        # random file order
        self.random_file_order()
        # send the first half to GPU
        self.send_to_device()
        self.firstHalf = True
        self.firstHalfData = 0
        # generate sample position
        self.sampleIndxMap = []
        self.noDataOnGPU = self.generateSamplePostionOnGPU()
        self.firstHalfData = self.noDataOnGPU
        print(self.filePage," files loaded. ", self.noData, " training examples loaded ", self.halfFile, " files sent to GPU, with total data number: ", self.noDataOnGPU)
        # shuffle the training sample index to get random batch
        self.shuffleSamplesOnGPU()

    def random_file_order(self):
        random.shuffle(self.filesIndex)

    def send_to_device(self, first_half=True):
        del self.EEG_on_GPU
        del self.Aud_on_GPU
        self.EEG_on_GPU = []
        self.Aud_on_GPU = []
        if first_half==True: # send the first half to GPU
            self.firstHalf = True
            for i in range(self.halfFile):
                tempEEG = self.EEGData[self.filesIndex[i]]
                tempAud = self.AudioData[self.filesIndex[i]]

                self.EEG_on_GPU.append(tempEEG.to(self.device))
                self.Aud_on_GPU.append(tempAud.to(self.device))

        else: # send the last half to GPU
            self.firstHalf = False
            for i in range(self.halfFile,self.filePage):
                tempEEG = self.EEGData[self.filesIndex[i]]
                tempAud = self.AudioData[self.filesIndex[i]]

                self.EEG_on_GPU.append(tempEEG.to(self.device))
                self.Aud_on_GPU.append(tempAud.to(self.device))

    def generateSamplePostionOnGPU(self):
        count = 0
        for i in range(len(self.EEG_on_GPU)):
            _,totalLength = self.EEG_on_GPU[i].shape
            startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
            for pos in startPos:
                self.sampleIndxMap.append((i, pos))
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            assert len(startPos) == noData
            count += noData
        return count

    def shuffleSamplesOnGPU(self):
        random.shuffle(self.sampleIndxMap)

    def convertToTensorType(self):
        self.EEGData = [torch.from_numpy(eegtrial).permute(1,0) for eegtrial in self.EEGData]
        self.AudioData = [torch.from_numpy(audiotrial).permute(1,0) for audiotrial in self.AudioData]

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

    def getNoData(self):
        count = 0
        for i in range(self.filePage):
            _,totalLength = self.AudioData[i].shape
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            count += noData
        return count

    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        if  self.firstHalf==True:
            if idx <= self.noDataOnGPU-1:
                fileIndex = self.sampleIndxMap[idx][0]
                endIndex = self.sampleIndxMap[idx][1]
                startIndex = self.sampleIndxMap[idx][1] - self.frameLength
                return self.EEG_on_GPU[fileIndex][:,startIndex:endIndex], self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
            else: # all data on GPU has just been iterated
                # send the second half to GPU
                self.send_to_device(first_half=False)
                # generate sample position
                self.sampleIndxMap = []
                self.noDataOnGPU = self.generateSamplePostionOnGPU()
                #print("first half data finished ", len(self.EEG_on_GPU)," files sent to GPU, with total data number: ", self.noDataOnGPU)
                # shuffle the training sample index to get random batch
                self.shuffleSamplesOnGPU()
                idx_new = idx - self.firstHalfData
                assert(self.noDataOnGPU == self.noData-self.firstHalfData)
                fileIndex = self.sampleIndxMap[idx_new][0]
                endIndex = self.sampleIndxMap[idx_new][1]
                startIndex = self.sampleIndxMap[idx_new][1] - self.frameLength
                return self.EEG_on_GPU[fileIndex][:,startIndex:endIndex], self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
        else:
            idx_new = idx - self.firstHalfData
            fileIndex = self.sampleIndxMap[idx_new][0]
            endIndex = self.sampleIndxMap[idx_new][1]
            startIndex = self.sampleIndxMap[idx_new][1] - self.frameLength
            if idx < self.noData-1:
                return self.EEG_on_GPU[fileIndex][:, startIndex:endIndex], self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
            else: # the last sample from one complete iteration
                tempEEG = self.EEG_on_GPU[fileIndex][:, startIndex:endIndex]
                tempAud = self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
                # reshuffle for a new iteration
                self.random_file_order()
                self.send_to_device(first_half=True)
                # generate sample position
                self.sampleIndxMap = []
                self.noDataOnGPU = self.generateSamplePostionOnGPU()
                self.firstHalfData = self.noDataOnGPU
                #print("finish one iteration ", len(self.EEG_on_GPU), " files sent to GPU, with total data number: ", self.noDataOnGPU)
                # shuffle the training sample index to get random batch
                self.shuffleSamplesOnGPU()
                return tempEEG, tempAud

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

    def send_to_device(self, device, no=1):
        if no==1:
            self.EEGData = [eegtrial.to(device)  for eegtrial in self.EEGData]
            self.AudioData = [audiotrial.to(device) for audiotrial in self.AudioData]
        else:
            noFilesToSend = math.floor(self.filePage*no)
            newEEGData = []
            newAudData = []
            for i in range(self.filePage):
                tempEEG = self.EEGData.pop(0)
                tempAud = self.AudioData.pop(0)
                if i<noFilesToSend:
                    newEEGData.append(tempEEG.to(device))
                    newAudData.append(tempAud.to(device))
                else:
                    newEEGData.append(tempEEG)
                    newAudData.append(tempAud)
            assert (len(newAudData)==self.filePage)
            self.EEGData = newEEGData
            self.AudioData = newAudData




    def convertToTensorType(self):
        self.EEGData = [torch.from_numpy(eegtrial).permute(1,0) for eegtrial in self.EEGData]
        self.AudioData = [torch.from_numpy(audiotrial).permute(1,0) for audiotrial in self.AudioData]

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
        fileIndex = self.sampleIndxMap[idx][0]
        endIndex = self.sampleIndxMap[idx][1]
        startIndex = self.sampleIndxMap[idx][1] - self.frameLength
        if self.EEGData[fileIndex].is_cuda:
            return self.EEGData[fileIndex][:,startIndex:endIndex], self.AudioData[fileIndex][:,startIndex:endIndex]
        else:
            return self.EEGData[fileIndex][:,startIndex:endIndex].cuda(), self.AudioData[fileIndex][:,startIndex:endIndex].cuda()


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