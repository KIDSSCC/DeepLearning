# import matplotlib.pyplot as plt
#
# # Define the vectors
# vector1 = [0.15390487601601194, 0.12308147681935411, 0.136033388903642, 0.11255608269082226, 0.1131949081462926, 0.11118228059929158, 0.10044866006695355, 0.10697343175649386, 0.10155182511138869, 0.09355187976274983]
# vector2 = [0.15082687241486095, 0.12446357893243137, 0.09659136663251087, 0.10200386420759847, 0.09154021402092605, 0.11176221832927684, 0.0857900644627161, 0.08063429032380157, 0.09297089407528909, 0.07900561881728754]
# vector3 = [0.18353133882059694, 0.11682009119179677, 0.101356082392065, 0.09515434719039872, 0.08937588580217813, 0.07812665514771629, 0.08007297423142291, 0.07454589208775227, 0.07809875152853142, 0.0739634132510233]
# vector4 = [0.35553913349255967, 0.24492415653381008, 0.19230967228815626, 0.15839013353008002, 0.13798652660286131, 0.1252849108941954, 0.11637339863990145, 0.10743357123650489, 0.10053455838500285, 0.09341407518316143]
# vector5 = [2.2663329264607293, 2.1954222121558633, 2.0561309782460855, 1.8196568729016727, 1.5419724257990195, 1.2637565206414976, 1.0767574666407163, 0.9529672254579136, 0.8535415675883857, 0.6815278512982134]
#
# # Create x-axis values (assuming indices as x-axis)
# x = range(len(vector1))
#
# # Plot the vectors
# plt.plot(x, vector1, label='0.2')
# plt.plot(x, vector2, label='0.1')
# plt.plot(x, vector3, label='0.05')
# plt.plot(x, vector4, label='0.01')
# plt.plot(x, vector5, label='0.0005')
#
# # Add labels and title
# plt.xlabel('epoch')
# plt.ylabel('Loss')
# plt.title('Loss with learning rate')
#
# # Add legend
# plt.legend()
#
# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import torch
#
# # Define the tensors
# tensor1 = torch.tensor([95.1500, 96.2200, 96.1200, 96.7500, 96.8500, 96.8500, 97.1700, 97.1500, 97.2900, 97.3900])
# tensor2 = torch.tensor([95.3300, 96.3700, 97.2300, 97.0000, 97.2500, 96.7200, 97.4900, 97.6800, 97.3100, 97.6000])
# tensor3 = torch.tensor([94.4500, 96.4400, 96.8300, 97.0500, 97.2700, 97.6400, 97.4700, 97.6300, 97.7800, 97.7700])
# tensor4 = torch.tensor([89.8400, 92.8700, 94.2100, 95.2300, 95.8000, 96.2100, 96.4700, 96.7400, 97.0700, 97.2700])
# tensor5 = torch.tensor([31.4900, 40.8900, 46.6000, 58.7100, 75.7500, 80.2100, 82.3400, 83.8900, 85.2000, 86.1900])
# # Create x-axis values (assuming indices as x-axis)
# x = range(len(tensor1))
#
# # Plot the tensors
# plt.plot(x, tensor1, label='0.2')
# plt.plot(x, tensor2, label='0.1')
# plt.plot(x, tensor3, label='0.05')
# plt.plot(x, tensor4, label='0.01')
# plt.plot(x, tensor5, label='0.0005')
#
# # Add labels and title
# plt.xlabel('epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy with learning rate')
#
# # Add legend
# plt.legend()
#
# # Show the plot
# plt.show()



# blocks=[6,12,24,16]
# for i in range(len(blocks)):
#     print(i)

import torch

# a=torch.ones(4,3,32,32)
# b=torch.ones(4,8,32,32)
# c=torch.cat([a,b],1)
# print(c.size())
import numpy as np
import matplotlib.pyplot as plt

a=[1,2,3,4,5]
plt.plot(np.arange(1,9+1,2), a)
plt.title('validation loss')
plt.show()