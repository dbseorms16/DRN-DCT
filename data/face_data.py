import os
from data import srdata


class face_data(srdata.SRData):
    def __init__(self, args, name='face_data', train=True, benchmark=False):
        super(face_data, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(face_data, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'face_data_HR')
        self.dir_lr = os.path.join(self.apath, 'face_data_LR_bicubic')

