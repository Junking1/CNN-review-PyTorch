import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as desets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image wight
LR = 0.1
DOWNLOAD_MNIST = False

train_data = desets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loder = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE, shuffle=True)

test_data = desets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,           #(batch, time_step, input)
        )
        self.out = nn.Linear(64, 10)


    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)          # x (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :])  # (batch,time_step, input)
        return out

rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loder):
        b_x = Variable(x.view(-1, 28, 28))   # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
