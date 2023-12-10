import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer


# class LSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.n_directions = 2 if bidirectional else 1
#         self.embedding = nn.Embedding(input_dim, 768)
#         self.lstm1 = nn.LSTM(768, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_dim * self.n_directions, output_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text):
#         embedded = self.dropout(self.embedding(text))
#         output, (hidden, cell) = self.lstm1(embedded)
#         hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
#         return torch.sigmoid(self.fc(hidden.squeeze(0)))
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask = None):
        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        K = keys.permute(*range(keys.dim() - 2), keys.dim() - 1, keys.dim() - 2)
        scores = torch.matmul(queries, K) / self.input_dim**0.5

        attention = torch.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        output = torch.matmul(attention, values)

        return output, attention


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query, self.key, self.value = [nn.Linear(input_dim, input_dim) for _ in range(3)]
        self.scaled_dot_product_attention = ScaledDotProductAttention(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        queries, keys, values = map(lambda linear_layer: linear_layer(x), (self.query, self.key, self.value))
        output, attention = self.scaled_dot_product_attention(queries, keys, values, mask=mask)
        output = self.dropout(output)

        return output

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_dim, 768)
        self.lstm1 = nn.LSTM(768, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        # self.additional_layers = nn.ModuleList([
        #     nn.LSTM(hidden_dim * self.n_directions, hidden_dim, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        #     for _ in range(n_layers - 1)
        # ])
        self.attention = SelfAttention(hidden_dim * self.n_directions)
        self.fc = nn.Linear(hidden_dim * self.n_directions, output_dim)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm1(embedded)
        # for layer in self.additional_layers:
        #     output, (hidden, cell) = layer(output)
        a_output = self.attention(output)
        c_output = output + a_output
#         c_output = self.relu(c_output)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return torch.sigmoid(self.fc(hidden.squeeze(0)))



def predict(input):
    model = LSTM(30522, 256, 1, 3, True, 0.5)
    model.load_state_dict(torch.load('LSTM/pickle_files/model_lstm_3.pt', map_location=torch.device('cpu')))
    device = torch.device('cpu')
    model.eval()
    max_len = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_text = input
    input = tokenizer.encode(input, max_length=max_len, truncation=True, padding=True)
    input = np.array(input)
    if len(input) > max_len:
        input = input[:max_len]
    elif len(input) < max_len:
        input = np.pad(input, (0, max_len-len(input)), 'constant')
    
    input = torch.tensor(input)
    input = input.unsqueeze(0)
    input = input.to(device)
    
    with torch.no_grad():
        output = model(input)
        output = output.squeeze()
        pred = torch.round(output)
        # 1 represents real news and 0 represents fake news
        if pred >= 0.5:
            print("Prediction for the given news {} is Real News".format(input_text))
            return "Real News"
        else:
            print("Prediction for the given news {} is Fake News".format(input_text))
            return "Fake News"


        
# using gradio take input from user and predict the output and display it to user
def fake_news_detection():
    # input_text = gr.inputs.Textbox(lines=10, label="Enter the news text here")
    # output_text = gr.outputs.Textbox(label="Prediction")
    interface = gr.Interface(fn=predict, inputs="text", outputs="text", 
                             title="Fake News Detection", description="Detects whether the news is real or fake")
    interface.launch(share=True)
    


fake_news_detection()

