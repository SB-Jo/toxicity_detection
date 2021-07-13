import torch.nn as nn
import torch

class RNNClassifier(nn.Module):
    def __init__(
        self,
        con_input_size,
        res_input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=.3
    ):
        self.con_input_size = con_input_size
        self.res_input_size = res_input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.con_emb = nn.Embedding(con_input_size, word_vec_size)
        self.res_emb = nn.Embedding(res_input_size, word_vec_size)
        self.con_rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )

        self.res_rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )


        self.generator = nn.Linear(hidden_size*4, n_classes)

        self.activation = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, con, res):
        # |x| = (batch_size, length(문장길이))
        con = self.con_emb(con)
        res = self.res_emb(res)

        # |con| = (batch_size, length, word_vec_size)
        con, _ = self.con_rnn(con)
        res, _ = self.res_rnn(res)

        x = torch.cat([con, res], dim=-1)
        # |x| = (batch-size, langth, hidden_size*4)

        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)
        return y