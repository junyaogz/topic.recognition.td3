#!/usr/bin/env python
# coding: utf-8

# In[601]:


# [Author]: Jun Yao
# [Date]: 2021-12-10

# [Description] 
# this file has the following functionalities
# (1) train model 1 in the paper and evaluate it against test data with golden labels.
# (2) calculate random guess accuracy
# (3) evaluate the decoded texts from model 2 (tri3 model trained in Kaldi).

# input of this script:
# stm_transcript_labels.csv
# test_text_from_model2.csv

# output of this script:
# prediction accuracy in the conclusion

# [Conclusion] 
# (1) random guess accuracy is merely 0.11, 
# (2) test accuracy of model 1 using the transcripts provided by TEDLIUM-3 is 0.40.
# (3) test accuracy of model 1 using the decoded text provided by model 2 is 0.28.
# as a reference, human prediction accuracy by the author is 0.53 (tried 3 times and pick the highest), 

# [References]
# 1. https://keras.io/examples/nlp/multi_label_classification/
# 2. https://en.wikipedia.org/wiki/Multi-label_classification


# In[602]:


from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[603]:


orig_data = pd.read_csv("stm_transcript_labels.csv",sep=",", error_bad_lines=False)
print(f"There are {len(orig_data)} rows in the dataset.")
orig_data.head()


# In[604]:


# ================ Remove duplicate items
total_duplicate_titles = sum(orig_data["titles"].duplicated())
print(f"There are {total_duplicate_titles} duplicate titles.")
orig_data = orig_data[~orig_data["titles"].duplicated()]
print(f"There are {len(orig_data)} rows in the deduplicated dataset.")
# There are some terms with occurrence as low as 1.
print(sum(orig_data["terms"].value_counts() == 1))
# How many unique terms?
print(orig_data["terms"].nunique())
# Filtering the rare terms.
orig_data_filtered = orig_data.groupby("terms").filter(lambda x: len(x) > 1)
orig_data_filtered.shape

# ================ Convert the string labels to lists of strings
orig_data_filtered["terms"] = orig_data_filtered["terms"].apply(lambda x: literal_eval(x))
orig_data_filtered["terms"].values[:5]

# ================ Use stratified splits because of class imbalance
test_split = 0.4
# Initial train and test split.
train_df, test_df = train_test_split(
    orig_data_filtered,
    test_size=test_split,
    stratify=orig_data_filtered["terms"].values,
)
# Splitting the test set further into validation
# and new test sets.
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)
print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")


# In[605]:


# ================ Multi-label binarization
terms = tf.ragged.constant(train_df["terms"].values)
#terms = tf.ragged.constant(orig_data_filtered["terms"].values)

lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()
def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)
print("Vocabulary:\n")
print(vocab)
sample_label = train_df["terms"].iloc[0]
print(f"Original label: {sample_label}")
label_binarized = lookup([sample_label])
print(f"Label-binarized representation: {label_binarized}")


# In[606]:


# ================ Data preprocessing and tf.data.Dataset objects
train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()


# In[607]:


max_seqlen = 2200
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE

def unify_text_length(text, label):
    # Split the given abstract and calculate its length.
    word_splits = tf.strings.split(text, sep=" ")
    sequence_length = tf.shape(word_splits)[0]

    # Calculate the padding amount.
    padding_amount = max_seqlen - sequence_length

    # Check if we need to pad or truncate.
    if padding_amount > 0:
        unified_text = tf.pad([text], [[0, padding_amount]], constant_values="<pad>")
        unified_text = tf.strings.reduce_join(unified_text, separator="")
    else:
        unified_text = tf.strings.reduce_join(word_splits[:max_seqlen], separator=" ")

    # The expansion is needed for subsequent vectorization.
    return tf.expand_dims(unified_text, -1), label

def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    dataset = dataset.map(unify_text_length, num_parallel_calls=auto).cache()
    return dataset.batch(batch_size)

# prepare the tf.data.Dataset objects
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)


# In[608]:


# ================ Dataset preview
text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text[0]}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")


# In[609]:


# ================ Vectorization
train_df["total_words"] = train_df["summaries"].str.split().str.len()
vocabulary_size = train_df["total_words"].max()
print(f"Vocabulary size: {vocabulary_size}")

text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

# `TextVectorization` layer needs to be adapted as per the vocabulary from our
# training set.
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)


# In[610]:


# ================ Create a text classification model
def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return shallow_mlp_model


# In[611]:


# ================ Train the model
epochs = 20

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("categorical_accuracy")


# In[612]:


# ================ Evaluate the model
_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")


# In[613]:


# Create a model for inference.
model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

print(test_df.shape)
print(test_df.iloc[0:5,:])

# Create a small dataset just for demoing inference.
inference_dataset = make_dataset(test_df.sample(5), is_train=False)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)
predicted_acc = 0 

# Perform inference.
for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    #print(f"Abstract: {text[0]}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    predicted_proba = [proba for proba in predicted_probabilities[i]]
    top_3_labels = [
        x
        for _, x in sorted(
            zip(predicted_probabilities[i], lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
    print(" ")
    
    predicted_acc = predicted_acc +     len(set(invert_multi_hot(label[0])).intersection([label for label in top_3_labels]))

print(f"number of correct labels is {predicted_acc}, prediction accuracy is {predicted_acc/15:.2f}") 


# In[614]:


# accuracy of random guess
import scipy.special
num_labels = 270
num_selected = 15
random_guess_accuracy = 0
for i in range(num_selected):
    a = i
    b = scipy.special.binom(num_selected,i)
    c = scipy.special.binom(num_labels - i, num_selected-i)
    d = scipy.special.binom(num_labels, num_selected)
    
    random_guess_accuracy = random_guess_accuracy + (a*b*c)/d
    
print(f"expected correct labels of random guess is merely {random_guess_accuracy:.2f}, accuracy is {random_guess_accuracy/num_selected:.2f}")  


# In[615]:


# Create the test dataset from decoded text of the kaldi model (model 2)
decode_test_df = pd.read_csv("test_text_from_model2.csv",sep=",", error_bad_lines=False)
decode_test_df_len = len(decode_test_df)

#print(decode_test_df.shape)
#print(decode_test_df)

decode_test_df["terms"] = decode_test_df["terms"].apply(lambda x: literal_eval(x))

def make_testdataset(dataframe):
    labels = tf.ragged.constant(dataframe["terms"].values)
    print(labels)
    label_binarized = lookup(labels).numpy()
    #print(dataframe.shape)
    #print(label_binarized)
    #print(label_binarized.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.map(unify_text_length, num_parallel_calls=auto).cache()
    return dataset.batch(batch_size)

inference_dataset = make_testdataset(decode_test_df)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)
predicted_acc = 0 

# Perform inference.
for i, text in enumerate(text_batch[:]):
    label = label_batch[i].numpy()[None, ...]
    #print(f"Abstract: {text[0]}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    predicted_proba = [proba for proba in predicted_probabilities[i]]
    top_3_labels = [
        x
        for _, x in sorted(
            zip(predicted_probabilities[i], lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
    print(" ")
    
    predicted_acc = predicted_acc +     len(set(invert_multi_hot(label[0])).intersection([label for label in top_3_labels]))

print(f"number of correct labels is {predicted_acc}, prediction accuracy is {predicted_acc/(decode_test_df_len*3):.2f}") 


# In[ ]:




