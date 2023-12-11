import gradio as gr
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math

# Transformer Model
# class TransformerModel(nn.Module):
#     def __init__(self, input_size=128, hidden_size=256, num_layers=3, num_heads=4, dropout_prob=0.1):
#         super(TransformerModel, self).__init__()

#         self.embedding_layer = nn.Linear(input_size, hidden_size)
#         self.attention_mask = nn.Linear(input_size, hidden_size)
        
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_size,
#                 nhead=num_heads,
#                 dropout=dropout_prob
#             ),
#             num_layers=num_layers
#         )

#         self.fc = nn.Linear(hidden_size, 1)  # Assuming binary classification (fake or not fake)

#     def forward(self, input_embeddings, attention_mask):
#         x = self.embedding_layer(input_embeddings)
#         x += self.attention_mask(attention_mask)
#         x = self.transformer_encoder(x)
#         x = self.fc(x)
#         x = torch.sigmoid(x)
#         return x


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Global Average Pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)


        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.global_avg_pooling(x.permute(0, 2, 1)).squeeze(2)
        output = self.fc(x)
#         output = torch.sigmoid(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.position_encoding = torch.tensor([[torch.sin(torch.tensor(pos / (10000.0 ** (i // 2 * 2.0 / d_model)))) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000.0 ** ((i) // 2 * 2.0 / d_model)))) for i in range(d_model)] for pos in range(max_len)])
        self.position_encoding = self.position_encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.position_encoding[:x.size(0), :].cuda()
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]        
        Query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(Query, Key.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention_scores, dim=-1)

        outputs = torch.matmul(attention, Value).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        outputs = self.fc_out(outputs)
        return outputs

class Feedforward(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout):
        super(Feedforward, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, pf_dim=512, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.feedforward = Feedforward(hidden_dim, pf_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attention_output = self.attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
    
    



def predict(input):
    columns=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']
    train_data = pd.read_csv('Dataset/train.tsv', sep='\t', header=None, names=columns,index_col=False)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = train_data['statement']
    train_transformed = vectorizer.fit_transform(X_train)
    input_transformed = vectorizer.transform([input]).toarray()
    input_transformed = torch.tensor(input_transformed, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_dim=5000, hidden_dim=128, output_dim=1, num_heads=4, num_layers=2)
    model.load_state_dict(torch.load('./attn_liar2.pt', map_location=device))
    model.eval()
    input_transformed = input_transformed.to(device)
    output = model(input_transformed.unsqueeze(0))
    pred = torch.sigmoid(output).item()
    # 1 represents real news and 0 represents fake news
    if pred >= 0.5:
        print("Prediction for the given news {} is Real News".format(input))
        return "Real News"
    else:
        print("Prediction for the given news {} is Fake News".format(input))
        return "Fake News"
    

        
# using gradio take input from user and predict the output and display it to user
def fake_news_detection():
    interface = gr.Interface(fn=predict, inputs="text", outputs="text", 
                             title="Fake News Detection", description="Detects whether the news is real or fake")
    interface.launch(share=True)
    


fake_news_detection()

