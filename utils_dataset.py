import h5py, torch, sys, os, sqlite3
import torch.utils.data.dataset

class HDF5Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, filename, collection_name='name'):
        self.f = h5py.File(filename,'r')
        self.dset = self.f[collection_name]

    def __getitem__(self, index):
        return self.dset[index]

    def __len__(self):
        return len(self.dset)

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
