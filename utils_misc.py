import torch, sys, os, numpy as np

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi')
    memory_available = [int(x.split()[2]) for x in open('tmp_smi', 'r').readlines()]
    os.remove("tmp_smi")
    return np.argmax(memory_available)

def cut300(text):
    return " ".join(text.split()[:300])

class DoublePrint(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.sterr = sys.stderr
        sys.stderr = self
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
