import os
from data import srdata


class div2k(srdata.SRData):
    def __init__(self, args, name='div2k', train=True, benchmark=False):
        super(div2k, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(div2k, self)._set_filesystem(data_dir)
        s = str(self.args.total_scale).split('.')
        
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('', '.png')

