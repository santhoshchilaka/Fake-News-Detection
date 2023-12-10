# Fake-News-Detection

In the digital age where fake news proliferates rapidly, our project explores the use of advanced NLP techniques for effective detection and differentiation of true vs. false narratives. With the rise of social media and the widespread dissemination of information, the spread of fake news has become a significant issue in society. Fake news often has characteristics such as sensational headlines, emotionally charged language, lack of credible sources, and confirmation bias. Our project aims to showcase the effectiveness and efficiency of both models in fake news detection, providing insights into their performance and potential applications in automated fact-checking. We harness the strengths of LSTM (Long Short-Term Memory) and Transformer models, renowned for their ability to process sequential data and large datasets, respectively, in the context of complex text analysis. We were able to get around 62\% accuracy for our models.

Models:

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

3. Transformer
 ![image](https://github.com/santhoshchilaka/Fake-News-Detection/assets/51093711/b50c148b-e525-47c8-a3a8-543aed5e0ad4)
