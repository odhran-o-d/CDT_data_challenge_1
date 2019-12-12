import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm
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

X_train = np.load('trans_x_train.npy', allow_pickle=True)
y_train = np.load('trans_y_train.npy', allow_pickle=True)
X_test = np.load('trans_x_test.npy', allow_pickle=True)
y_test = np.load('trans_y_test.npy', allow_pickle=True)

######## Network arch #######
class ConvBNReLU(nn.Module):
    ''' Convolution + batch normalization + ReLU is a common trio '''
    def __init__(
        self, in_channels, out_channels,
        kernel_size=5, stride=1, padding=1, bias=True
    ):
        super(ConvBNReLU, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                kernel_size, stride, padding, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main(x)


class Combine(nn.Module):
    def __init__(self, num_hn=128, num_class=5, grad_clip=1.0):  # num_hn=128
        super(Combine, self).__init__()
        cnnoutputsize = 191616  # 234
        self.conv1 = ConvBNReLU(3, 32)
        self.conv2 = ConvBNReLU(32, 32)
        self.conv3 = ConvBNReLU(32, 64)
        self.grad_clip = grad_clip

        self.rnn1 = nn.LSTM(
            input_size=cnnoutputsize,
            hidden_size=128,
            batch_first=True)
        self.rnn2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True)
        self.linear = nn.Linear(64, num_class)

    def forward(self, x):
        batch_size, timesteps, C, W = x.size()

        c_in = x.view(batch_size * timesteps, C, W)
        x = self.conv1(c_in)
        x = self.conv2(x)
        x = self.conv3(x)

        r_in = x.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn1(r_in)
        r_out, (h_n, h_c) = self.rnn2(r_out)

        r_out2 = self.linear(r_out)
        return F.log_softmax(r_out2, dim=1)

### Feed data

num_filters_init = 8  # initial num of filters -- see class definition
in_channels = 3  # num channels of the signal -- equal to 3 for our raw triaxial timeseries
output_size = 1  # number of classes (sleep, sedentary, etc...)
num_epoch = 20  # num epochs (full loops though the training set) for SGD training
lr = 1e-3  # learning rate in SGD
batch_size = 1  # size of the mini-batch in SGD

torch.cuda.empty_cache()
cnnlstm = Combine().to(device)
cnnlstm.double()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnnlstm.parameters(), lr=lr, amsgrad=True)
print(cnnlstm)


### Data loading
def create_dataloader(X, y=None, batch_size=1, shuffle=False):
    ''' Create a (batch) iterator over the dataset. Alternatively, use PyTorch's
    Dataset and DataLoader classes -- See
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html '''
    max_len = 1722
    if shuffle:
        idxs = np.random.permutation(np.arange(len(X)))
    else:
        idxs = np.arange(len(X))
    for i in range(0, len(idxs), batch_size):
        idxs_batch = idxs[i:i + batch_size]
        X_batch = X[idxs_batch]

        seq_tensor = Variable(torch.zeros((batch_size, max_len, 3, 3000))).double()
        seq_lengths = torch.LongTensor(list(map(len, X_batch)))

        for idx, (seq, seqlen) in enumerate(zip(X_batch, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # X_batch = torch.from_numpy(seq_tensor)

        if y is None:
            yield seq_tensor
        else:
            y_batch = y[idxs_batch]
            y_tensor = Variable(torch.zeros((batch_size, max_len))).long()
            for idx, (seq, seqlen) in enumerate(zip(y_batch, seq_lengths)):
                y_tensor[idx, :seqlen] = torch.LongTensor(seq)
            yield seq_tensor, y_tensor


def forward_by_batches(model, X, Y):
    ''' Forward pass model on a dataset. Do this by batches so that we do
    not blow up the memory. '''
    Y_hat = []
    true_Y = []
    model.eval()
    with torch.no_grad():
        for x, target in create_dataloader(X, Y, batch_size=1, shuffle=False):  # do not shuffle here!
            x = x.to(device)
            target = target.to(device)
            Y_hat.append(model(x))

            true_Y.append(target)
    model.train()
    Y_hat = torch.cat(Y_hat)
    true_Y = torch.cat(true_Y)
    return Y_hat.view(-1, 5), true_Y.view(-1)


def evaluate_model(model, X, y):
    max_len = 1722
    Y_pred, true_Y = forward_by_batches(model, X, y)  # scores
    print(Y_pred.size())
    print(true_Y.size())

    loss = F.cross_entropy(Y_pred, true_Y).item()
    Y_pred = F.softmax(Y_pred, dim=1)  # convert to probabilities
    y_pred = torch.argmax(Y_pred, dim=1)  # convert to classes
    y_pred = y_pred.cpu().numpy()  # cast to numpy array
    scores = utils.compute_scores(true_Y.cpu().numpy(), y_pred)
    return loss, scores


accuracy_history = []
balanced_accuracy_history = []
kappa_history = []
loss_history = []
loss_history_train = []
batch_size = 1
for i in tqdm(range(num_epoch)):
    dataloader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    losses = []
    for x, target in dataloader:
        x, target = x.to(device), target.to(device)
        cnnlstm.zero_grad()

        output = cnnlstm(x)

        loss = loss_fn(output.view(-1, 5), target.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(cnnlstm.parameters(),
                                 cnnlstm.grad_clip if cnnlstm.grad_clip > 0 else float('inf'))

        optimizer.step()
        # Logging -- track train loss
        losses.append(loss.item())

    # -------------------------------------------------------------------------
    # Evaluate performance at the end of each epoch (full loop through the
    # training set). We could also do this at every iteration, but this would
    # be very expensive since we are evaluating on a large dataset.
    # Aditionally, at the end of each epoch we train a Hidden Markov Model to
    # smooth the predictions of the CNN.
    # -------------------------------------------------------------------------

    # Logging -- average train loss in this epoch
    loss_history_train.append(np.mean(losses))

    loss_test, scores_test = evaluate_model(
         cnnlstm, X_test, y_test
    )
    loss_history.append(loss_test)
    accuracy_history.append(scores_test['accuracy'])
    balanced_accuracy_history.append(scores_test['balanced_accuracy'])
    kappa_history.append(scores_test['kappa'])


torch.save(cnnlstm.state_dict(), 'cnnlstm.mdl')

# Loss history
fig, ax = plt.subplots()
ax.plot(loss_history_train, color='C0', label='train')
ax.plot(loss_history, color='C1', label='test')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend()
fig.show()
plt.savefig('loss.png')

# Scores history
fig, ax = plt.subplots()
ax.plot(accuracy_history, label='accuracy')
ax.plot(balanced_accuracy_history, label='balanced accuracy')
ax.plot(kappa_history, label='kappa')
ax.set_ylabel('score')
ax.set_xlabel('epoch')
ax.legend()
fig.show()
plt.savefig('score.png')

# Scores details -- last epoch
utils.print_scores(scores_test)
