[Author]:  Jun Yao

[Date]:       2021-12-14

[Project Title]: Topic Recognition Using TED-LIUM Release 3

[Project Summary]: 
Multple topics recognition is a hard problem, but has important applications in industry.
For video sharing website, it is not easy to categorize user-uploaded videos timely and correctly, using a multple-topics recognition process the author created, administrator of the website can reduce number of times needed to manually add the labels/categories.
I created two models,
Model 1 is trained in tensorflow 2.0, it is text based using two layers of neural network, it can predict multiple topics based on the transcript provided. It is trained using the official transcipt and the golden topic labels I created. Creating the labels is time-consuming but very important for this study.
Model 2 is trained in Kaldi, using steps 1-15 of the offcial tedlium3 egs script, I uses only tri3 model since steps 1-15 doesn't need any GPU. The decoded text is the transcipt of the audios.
The decoded text from model 2 is then applied in model 1 to predict its corresponding topics.
Here is the result of the experiment:
(1) random guess accuracy is merely 0.11, 
(2) test accuracy of model 1 using the official transcripts provided by TEDLIUM-3 is 0.40.
(3) test accuracy of model 1 using the decoded text from model 2 is 0.28.
As a reference, human prediction accuracy by the author is 0.53 (tried 3 times and pick the highest), 

[Cautions]:
model 2 is quite time-consuming, takes up to 5 days to train in a google cloud VM (8C 32G)
model 1 takes many memeory, at least 8GB is needed to avoid peculiar problems (ocurred in google colab)
Data preparation is also time-consuming, the author uses 8 days to prepare the neccessay data including the labels.

[List of tools that were used]:
(1) Ubuntu 18.04 LTS: http://releases.ubuntu.com/18.04/
(2) Kaldi: https://github.com/kaldi-asr/kaldi
(3) Anaconda and Jupyter Notebook: https://www.anaconda.com/products/individual
(4) Tensorflow 2.0:  https://www.tensorflow.org/install
(5) Python 3: https://www.python.org/downloads/

[List of directories and executables that may be used to test the code]:
No extra executables needed, just run or follow the instructions in each folder.

[Instructions to run the scripts]
I split the whole process into five phases, each one is put in a separate folder:
1_create_golden_topic_labels
2_prepare_model1_data
3_train_model2_in_kaldi
4_transform_decoded_text_from_model2
5_train_and_test_model1_in_tf

Here are the explanations for each folder:

(1) 1_create_golden_topic_labels
run the .ipynb or .py, they are identical in code.
this script runs for about 36 hours.
.html file is the snapshot of my experiment, you can directly check this file 
if you don't want to run the time-consuming script.
stm folder is the official transcripts of the 2351 audios in TEDLIUM v3 dataset.
tedlium3_topic_labels.csv is the output.
video_uris.txt and video_uris_repaired.csv are intermediate files.

(2) 2_prepare_model1_data
run the .ipynb or .py, they are identical in code.
this script runs for about 1 minute.
.html file is the snapshot of my experiment, you can directly check this file 
if you don't want to run the script.
stm folder is the transcripts of the 2351 audios in TEDLIUM v3 dataset.
tedlium3_topic_labels.csv and stm folder are the input
stm_transcript_labels.csv is the output

(3) 3_train_model2_in_kaldi
instructions_to_train_model2.md is the instruction to of this phase.
I didn't provide a all-in-one script because when I was running the official egs
script, it cannot succeed in one time, it's better to split it into small steps 
and check each step carefully.
log folder is the output of the tri3 model.
run.log is the log of my experiment, you can directly check this file 
if you don't want to run the time-consuming script.
wer_17 is the result of WER and SER of the tri3 model.

(4) 4_transform_decoded_text_from_model2
run the .ipynb or .py, they are identical in code.
this script runs for about 1 minute.
.html file is the snapshot of my experiment, you can directly check this file 
if you don't want to run the script.
log folder is the output of previous phase and the input of current phase.
tedlium3_topic_labels.csv from phase 1 is also an input.
test_text_from_model2.csv is the output.

(5) 5_train_and_test_model1_in_tf
run the .ipynb or .py, they are identical in code.
this script runs for about 2 minute.
.html file is the snapshot of my experiment, you can directly check this file 
if you don't want to run the script.
stm_transcript_labels.csv and test_text_from_model2.csv are the input
this script output the prediction accuracy.

[Future Directions]
Due to the limited time and unfamiliarity of details of Kaldi, the author has many ideas that are not carried out yet.
Here are some possible directions The author want to try in the future:
(1) add more data, 2351 videos is enough for audio training, but is not enough for multiple labels training, 
    especially for a large pool of possible labels (more than 270 labels in the dataset).
(2) improve the quality of decoded text from kaldi text+acoustic model (model 2), i.e. improve WER and SER.
(3) recognize the correct start and end of each sentence, add correct punctuations to sentences.
(4) decrease the number of possible labels, e.g. from 270 to 30 or so.
(5) create end-to-end process which uses acoustic features and neural network only.
