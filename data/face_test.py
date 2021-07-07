import os
from data import srdata


class face_test(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(face_test, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'benchmark', self.name)
        s = str(self.args.total_scale).split('.')
        hr_path = 'HR_' + s[0] + s[1]
        lr_path = 'LR_' + s[0] + s[1]
        
        # self.dir_hr = os.path.join(self.apath, 'HR')
        # self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.dir_hr = os.path.join(self.apath, hr_path)
        self.dir_lr = os.path.join(self.apath, lr_path)
        self.ext = ('', '.jpg')

