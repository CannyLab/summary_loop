import torch.utils.data.dataset
import h5py, torch, sys, os
import numpy as np, sqlite3

class HDF5Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, filename, collection_name='name'):
        self.f = h5py.File(filename,'r')
        self.dset = self.f[collection_name]

    def __getitem__(self, index):
        return self.dset[index]

    def __len__(self):
        return len(self.dset)

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi')
    memory_available = [int(x.split()[2]) for x in open('tmp_smi', 'r').readlines()]
    os.remove("tmp_smi")
    return np.argmax(memory_available)

class SQLDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, filename, table_name='articles', cut=None):
        self.table_name = table_name
        self.conn = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self.cut = cut
        self.curr = self.conn.cursor()

    def __getitem__(self, index):
        if self.cut is not None:
            res = self.curr.execute("SELECT * FROM "+self.table_name+" WHERE cut_id=? and cut=?", (index, self.cut))
        else:
            res = self.curr.execute("SELECT * FROM "+self.table_name+" WHERE id= ?", (index,))
        return [dict(r) for r in res][0]

    def __len__(self):
        if self.cut is not None:
            N = self.curr.execute("SELECT COUNT(*) as count FROM "+self.table_name+" WHERE cut = ?", (self.cut,)).fetchone()[0]
        else:
            N = self.curr.execute("SELECT COUNT(*) as count FROM "+self.table_name).fetchone()[0]
        return N

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