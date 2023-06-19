from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import numpy as np
writer = SummaryWriter("logs")

# for i in range(100):
#     writer.add_scalar("y=x*x",i*i,i)

face_dir = "C:\\Users\\sugao\\Desktop\\研究生相关\\pytorch学习\\faces"
face_list = os.listdir(face_dir)
for i in range(len(face_list)):
    face_path = os.path.join(face_dir,face_list[i])
    image_PIL = Image.open(face_path)
    image_np = np.array(image_PIL)
    print(image_np.shape)
    writer.add_image("情绪变化",image_np,i,dataformats='HWC')

writer.close()

