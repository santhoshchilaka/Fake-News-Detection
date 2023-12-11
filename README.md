# Fake-News-Detection

In the digital age where fake news proliferates rapidly, our project explores the use of advanced NLP techniques for effective detection and differentiation of true vs. false narratives. With the rise of social media and the widespread dissemination of information, the spread of fake news has become a significant issue in society. Fake news often has characteristics such as sensational headlines, emotionally charged language, lack of credible sources, and confirmation bias. Our project aims to showcase the effectiveness and efficiency of both models in fake news detection, providing insights into their performance and potential applications in automated fact-checking. We harness the strengths of LSTM (Long Short-Term Memory) and Transformer models, renowned for their ability to process sequential data and large datasets, respectively, in the context of complex text analysis. We were able to get around 62\% accuracy for our models.

# Models:

1. Bi-LSTM

![image](https://github.com/santhoshchilaka/Fake-News-Detection/assets/51093711/a2b08bfc-6c8a-4435-8e8c-c900cc041bae)

The architecture presented in the diagram provides a structured approach to natural language pro-
cessing tasks, specifically designed for classifying text inputs. Initially, the raw text is subjected to a
series of preprocessing steps: the data is loaded, cleansed of noise such as unwanted characters and
spaces, and then tokenized using BERT’s sophisticated tokenization process. This step is essential
for breaking down the text into a form that a neural network can work with. Following tokenization,
label encoding is applied to translate categorical labels into numerical values, preparing the data for
the machine learning model.

Once preprocessed, the text data enters the deep learning phase of the pipeline. An embedding
layer first converts the tokenized text into dense vectors that are capable of capturing complex word
relationships and semantic meanings. These vectors then pass through multiple LSTM layers, which
are specifically chosen for their ability to process sequences and capture temporal dependencies. The
LSTM layers, potentially numerous as indicated by "N layers", allow the model to learn from the
context within the text over different ranges, from immediate succession to long-range patterns.

Finally, the outputs from the LSTM layers are concatenated to form a holistic feature set, which
is then refined through a dropout layer to prevent overfitting a common challenge in deep learning
models. The concatenated and regularized features feed into a fully connected (FC) layer, which
serves to distill these features into a format that can be used for classification. The process concludes
with a sigmoid function in the output layer, providing a probability score between 0 and 1. This
score is the model’s prediction of the likelihood that the input text belongs to a particular class, thus
achieving the classification objective

2. Bi-LSTM(attention)

![image](https://github.com/santhoshchilaka/Fake-News-Detection/assets/51093711/cb8de268-f21d-402c-b8e1-fff560ce9fc7)

The architecture encapsulates a sequence-to-sequence neural network designed for the classification
of text data. Starting with raw text input, the data is preprocessed through steps of loading, cleaning,
BERT tokenization, and label encoding, transforming the text into a format suitable for machine
learning. An embedding layer then assigns vectors to the tokenized text, which are processed
by multiple bidirectional LSTM layers. Three stacked bidirectional LSTM layers with a hidden
dimension of 256 facilitate comprehensive temporal understanding, extracting features from both
past and future states are augmented by a self-attention mechanism that assigns varying levels of
importance to different parts of the input sequence, enhancing the model’s ability to understand
context and relationships within the data.

Following the sequential processing, the concatenated outputs from the LSTM layers, rich with
contextual information, are subjected to a ReLU activation function, introducing non-linearity to the
feature set. A dropout layer is then applied to prevent overfitting by randomly disabling a subset
of neurons during training. The network’s culmination is a fully connected layer that integrates
these features into a final output, which is subsequently passed through a sigmoid activation function.
This final step produces a probability score, denoting the likelihood of the input text belonging to a
particular class, which finalizes the binary classification process. The architecture is thus tailored to
extract and utilize the intricate patterns within text data, enabling accurate classification decisions.


3. Transformer
   
This Transformer-based model begins with text data that is methodically cleaned and vectorized using TF-IDF, ensuring that the most significant words within the text are emphasized numerically. The model then embeds these vectors into a higher-dimensional space and integrates positional encoding, a critical component that allows the transformer to understand the order and relevance of words in a sequence. The self-attention mechanism within the Transformer block is pivotal, enabling the model to dynamically assign importance to different segments of the input data, a process that is inherently more global compared to the local focus in LSTM architectures.

Transformer blocks, employing multi-head attention and feedforward layers, capture intricate contextual dependencies, allowing the model to discern the relevance of different words within a statement. The inclusion of layer normalization stabilizes training, mitigating issues like vanishing or exploding gradients. Global average pooling reduces spatial dimensions, and the fully connected layer produces the final output for binary classification. The softmax or sigmoid output layer provides the probability scores for each class, giving a quantifiable measure for classification decisions. Unlike the sequential focus of LSTM models, this architecture leverages the parallel nature of Transformers to efficiently handle long-range dependencies and complex patterns in text data, making it highly suitable for sophisticated natural language processing tasks.

![image](https://github.com/santhoshchilaka/Fake-News-Detection/assets/51093711/b50c148b-e525-47c8-a3a8-543aed5e0ad4)

4. Bert

We just implemented this pre-trained Bert model just to compare our model results with this model. BERT is the first representation model based on fine-tuning that outperforms numerous task-specific designs and reaches state-of-the-art performance on a wide range of sentence and token-level problems. Here, we have two stages to complete which are pre-training and fine-tuning. The model was tested on unlabelled data across several pre-training tasks during pre-training. The pre-trained parameters are used to initialize the BERT model for fine-tuning, and labeled data from the downstream jobs is used to adjust every parameter. Even the Bert model just got around 65% accuracy which is not that much higher than our models, so we can say that our models are working well.


# Web Application:

We used Gradio to integrate our model into a web application where users can test news articles for authenticity. Run fake_test.py/attn_liar2_app.py to open the web interface through the link(url) generated.

![WhatsApp Image 2023-11-23 at 3 43 31 PM](https://github.com/santhoshchilaka/Fake-News-Detection/assets/59920639/b4df485f-3145-4f2f-8239-992e0dab0398)


# References:

1. https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
2. https://nshrimali21.medium.com/language-translation-transformers-attention-pytorch-35641c056992
3. https://github.com/youngbin-ro/Attention-Based-BiLSTM
4. https://medium.com/@skillcate/detecting-fake-news-with-a-bert-model-9c666e3cdd9b
5. https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a
6. https://spotintelligence.com/2023/01/31/self-attention/
7. https://github.com/SindhuMadi/FakeNewsDetection/tree/main
8. https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
9. https://github.com/Nish-19/BERT\_Tutorial/tree/main
10. https://github.com/JaySuthar/FAKE\_NEWS\_DETECTION\_USING\_CNN\_LSTM\_BILSTM\_BERT\_ROBERTA
