# Sotiris-Legkas

## Task selection
There were many options for plots in presantation, like tweets per name, mean sentiment score per name, mean sentiment score per location or through time (day-month). Also, there were many options for a highly interactive dashboard like a map (latitude,longitude) of the Netherlands with each the locations and a colour scale based on the sentiment score, which could changed interaactively for different dates of the year.

After carefully reviewed the tasks, I chose to build a practical ML model for sentiment analysis.

# Initial thoughts
I kept the English translation of the full text and the sentiment score (sentiment_pattern). For this task, I should have started with something simple like using a simple model like logistic regression or even a relatively simple MLP and then try something more complex like RNN or a pre-trained model. Also, I should have preprocess the corpus like remove html tags, remove punctuations, remove numbers, single characters, multiple spaces, stopping words and even use lemmatizer, strip accents and lowercase the words. Then I should have tokenized the text make embeddings and use them as input in my model. (Maybe use TFIDF and maybe SVD for dimansionality reduction).

However, I decided due to time constraints to go straight to pre-trained transformer models. I used the RoBERTa-base model. Hence, I used a pretrained model and fine-tuned it for this specific task, by training a top layer for some epochs.

Note: I should have used a multilingual model like XLM-RoBERTa in dutch text. I did not do it due to time constraints. However, the procedure is the same.

## Detailed project description
I used Google colab to utilize the GPU.

After creating a single DataFrame with the whole dataset, I had to check the basics in the dataset and make the necessary data preparation.

First, I plotted the missing values for each column to see the significance of the null values. Then I dropped the rows with missing text.

Second, I visualized the distribution of the sentiment value across bins with 0.1 width.
Based on that, I chose the threshold that I would transform the sentiment score from float [-1:1] to integer.
Based on the distribution:
* negative (0) : from -1 to 0
* neutral (1): from 0 to 0.15
* positive (2): from 0.15 to 1

Next, I split the dataset into train (80%) and validation (20%) and again the train into train (80%) and test (20%).
I tranformed the DataFrame into dataset type from Hugging face for easier manipulation and use.
Finally, I kept only a portion of the datasets for practical reasons. (2000 in train, 200 in validation and 200 in test)

Again, I checked the destribution  of the created hard labels in the whole dataset, which seemed as intented.

I loaded the tokenizer and the model of RoBERTa model along with thier configuration and created a preprocess function, which I applied in the datasets.

Note: It is important that train-validation-test sets keep almost the same sentiment distribution, which is the case.

I defined a function that computes the metrics of accuracy, macro F1, micro F1 during the training.

I initiallized the training arguments and the trainer

Note: I used early stopping with patience 3 epochs and trained it for 5 epochs max due to time contraints (There may be a need for further training ). Many other hyperparameters can be analyzed and discussed in a call meeting.

I trained the model, saved the logits and the results into CSV files. visualized the distribution of the predicted and true sentiments and finally plotted a confusion matrix.


-- | precision |   recall  |f1-score |  support
------------- | -------------|-------------|-------------|-------------
negative    |   0.55 |     0.30  |    0.39  |      93
neutral     |  0.70    |  0.82    |  0.76   |    228
positive    |   0.50    |  0.51    |  0.50   |     79
accuracy   |           |         |    0.64  |     400
macro avg       |0.58|      0.54 |     0.55  |     400
weighted avg     |  0.62    |  0.64    |  0.62   |    400

Note: The results were not ideal. It was just an indicative procedure to a solution. Also, the validation loss seemed off. However, there may be expanation.

Note 2: Probably, I should have tackled this task like a regression type problem, were the lowest sentiment score would be the negative and the highest the positive. To do this, I could have changed the compute_metrics function like using MAE and use as num_labels=1.

## For further explanation, I would be happy to discuss my solution in a call.
