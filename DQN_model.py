import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # format is "nn.Linear(<input>, <output>)
        # the <output> then serves as the next <input> for the next layer within the NN
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './LinearQNet_model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        # format is "os.path.join(<folder directory/path>, <saved file name>)"
        file_name = os.path.join(model_folder_path, file_name)
        #.state_dict() saves the NN_Model's Params
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() #loss fcn

    def train_step(self, state, action, reward, next_state, done):
        #converts from PyTorch to Tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # NOTE: dont need to create a conversion for "done" so it's ONLY  a single value

        #to handle multiple sizes
        if len(state.shape)==1:   #only 1-D (1, x) - w/ 1 = batch_size and 'x' dims via Flatten
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            #now we need to convert "done" from a single value to a tuple
            done = (done, )

        # 1: predicted Q values w/ current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx] #Q_new is the reward of the current idx
            if not done[idx]:
                Q_new = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))

            #converts target state to a value and not a tensor b/c tensors have no actionable magnitude
            target[idx][torch.argmax(action).item()] = Q_new


        #2: Q_new = R + gamma*[max(next_predicted_Q_value)] - Bellman Equation -> only do this if NOT Done
        # pred.clone() - will have 3 values i.e. [1,0,0] and the index '1' which is the action will become the input
        # for the "argmax(action)" and the Q_new state
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        _loss = self.criterion(target, pred)
        _loss.backward() #enables backprop

        self.optimizer.step()











