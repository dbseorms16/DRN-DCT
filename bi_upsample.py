import cv2

import os
import torch.nn as nn



scale = 4
# upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
root_dir = 'upsample_target'
target_dir = 'x' + str(scale)

file_path = os.path.join(root_dir, target_dir)
result_path = os.path.join("upsample_result", target_dir)
file_list = os.listdir(file_path)
os.makedirs(result_path, exist_ok = True)

for num in range(len(file_list)):
    file_name = os.path.join(file_path, file_list[num])
    result_file_name =file_list[num]
    result = os.path.join(result_path, result_file_name )

    x =cv2.imread(file_name)
    x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(result, x)
    print(file_list[num] + ' >> ' +result_file_name +'  Completed!')

    


