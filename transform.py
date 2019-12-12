import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# For reproducibility
import utils


np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = True

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))


data = np.load('capture24.npz', allow_pickle=True)
print("Contents of capture24.npz:", data.files)
X_feats, y, pid, time, annotation = \
    data['X_feats'], data['y'], data['pid'], data['time'], data['annotation']
print('X_feats shape:', X_feats.shape)
print('y shape:', y.shape)
print('pid shape:', pid.shape)
print('time shape:', time.shape)
print('annotation shape:', annotation.shape)
X_raw = np.load('X_raw.npy', mmap_mode='r')
print('X_raw shape:', X_raw.shape)
data_unl = np.load('capture24_test.npz')
print("\nContents of capture24_test.npz:", data_unl.files)


def interval2seq(mydata):
    X_feats, y, pid, time, annotation = \
        mydata['X_feats'], mydata['y'], mydata['pid'], mydata['time'], mydata['annotation']

    # Get all the unique pids
    subjects = np.unique(mydata['pid'])
    X_tr = []
    y_tr = []
    pid_tr = []
    for subject in subjects:
        if subject == 139:
            continue
        sub_raw = X_raw[pid == subject]
        y_row = y[pid == subject]

        X_tr.append(np.split(sub_raw, [int(len(sub_raw) / 3), int(len(sub_raw) * 2 / 3)])[0])
        X_tr.append(np.split(sub_raw, [int(len(sub_raw) / 3), int(len(sub_raw) * 2 / 3)])[1])
        X_tr.append(np.split(sub_raw, [int(len(sub_raw) / 3), int(len(sub_raw) * 2 / 3)])[2])

        y_tr.append(np.split(y_row, [int(len(y_row) / 3), int(len(y_row) * 2 / 3)])[0])
        y_tr.append(np.split(y_row, [int(len(y_row) / 3), int(len(y_row) * 2 / 3)])[1])
        y_tr.append(np.split(y_row, [int(len(y_row) / 3), int(len(y_row) * 2 / 3)])[2])

        pid_tr.append(subject)
        pid_tr.append(subject)
        pid_tr.append(subject)

    x = np.array(X_tr)
    y = np.array(y_tr)
    pid_tr = np.array(pid_tr)

    return x, y, pid_tr

x, y, pid = interval2seq(data)

seq_lengths = torch.LongTensor(list(map(len, x)))
seq_lengths
seq_tensor = Variable(torch.zeros((len(x), seq_lengths.max()))).long()

seq_tensor.size()

# Hold out some participants for testing the model
pids_test = [2, 3]
mask_test = np.isin(pid, pids_test)
mask_train = ~mask_test
y_train, y_test = y[mask_train], y[mask_test]
pid_train, pid_test = pid[mask_train], pid[mask_test]
# X[mask_train] and X[mask_test] if you like to live dangerously
X_train = utils.ArrayFromMask(x, mask_train)
X_test = utils.ArrayFromMask(x, mask_test)

np.save('trans_x_train', X_train)
np.save('trans_y_train', y_train)
np.save('trans_x_test', X_test)
np.save('trans_y_test', y_test)
