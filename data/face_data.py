import os
from data import srdata


class face_data(srdata.SRData):
    def __init__(self, args, name='face_data', train=True, benchmark=False):
        super(face_data, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(face_data, self)._set_filesystem(data_dir)
        s = str(self.args.total_scale).split('.')
        hr_path = 'HR_' + s[0] + s[1]
        lr_path = 'LR_' + s[0] + s[1]
        
        # self.dir_hr = os.path.join(self.apath, 'HR')
        # self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.dir_hr = os.path.join(self.apath, hr_path)
        self.dir_lr = os.path.join(self.apath, lr_path)

