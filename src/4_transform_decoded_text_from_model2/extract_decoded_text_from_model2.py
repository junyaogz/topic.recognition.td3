#!/usr/bin/env python
# coding: utf-8

# In[76]:


# [Author]: Jun Yao
# [Date]: 2021-12-03

# [Description] 
# this file transformed decoded text(transcript) from model 2, it will be used as test data in model 1
# as a comparison to text created from the official transcript.

# [Instructions] 
# decoded text is stored in the log files of tri3 model from kaldi.
# so log files from model 2 is needed to do this transformation.
# the author copied the log files into the same folder where this script sits

# input of this script:
# ./log/decode.*.log
# tedlium3_topic_labels.csv

# output of this script:
# test_text_from_model2.csv


# In[77]:


import pandas as pd

#log files come from:
#kaldi-trunk/egs/tedlium/s5_r3/exp/tri3/decode_test/log/decode.*.log
#they will be copied to ./log/ after kaldi training and decoding process

tri3_decode_path = "./log/" 

# read gold labels file
df = pd.read_csv("tedlium3_topic_labels.csv")

# add a new column to the dataframe
df = df.assign(TRANSCRIPT="")

print(len(df))
df.head()


# In[78]:


import re
import os

def read_decoded_text_from_kaldi(file_name):
    #print(file_name)
    full_path = tri3_decode_path + file_name
    #print(f"file path: {full_path}")
    trans_name = None
    trans = []
    with open(full_path) as file:
        for line in file:
            if line.startswith("LOG"):
                continue
            m = re.findall(r"\d{7}-\d{7}", line)
            if m:
                #print(line)
                #print(f"find segement: {m[0]}")
                splitstrs = line.split("-"+m[0])
                trans_name = splitstrs[0]
                trans.append(splitstrs[1].replace("<unk>","").replace("'","").replace("\n","").strip())
    return [trans_name,". ".join(trans)]

list_of_files = []

for root, dirs, files in os.walk(tri3_decode_path):
    for file in files:
        if file.startswith("decode."):
            list_of_files.append(file)
print(f"====>find {len(list_of_files)} log files:")
for name in list_of_files:
    print(name)

df_len = len(df)

# read all decoded texts
for log_file_name in list_of_files:
    print(f"====>finding decoded segements in {log_file_name}....")
    with open(tri3_decode_path+log_file_name) as file:
        tmp = read_decoded_text_from_kaldi(log_file_name)
        print(f"====>decoded text from {log_file_name}:")
        #print(tmp)
        if tmp[0] is None:
            continue
            
        # add transcript to df
        fname = tmp[0]+".stm"
        print(f"stm file name is {fname}")
        year = tmp[0].split("_")[1]
        if len(year)>4:
            year = year[0:4]
        file_name_to_match = tmp[0].split("_")[0]+"_"+str(year)
        print(file_name_to_match)
        for i in range(df_len):
            #print(df.iloc[i,1])
            if df.iloc[i,1].find(tmp[0].split("_")[0])>-1:
                print("found")
                df.iloc[i,10] = df.iloc[i,10] + ". " + tmp[1]
                break
        #break
    
print("success")

filtered_df = df[df['TRANSCRIPT'] != ""]
print(f"=====>found {len(filtered_df)} files, final dataframe is:")
print(filtered_df)


# In[85]:


# prepare the data for comparison in model 1
column_names = ["titles", "summaries", "terms"]
df_md_1 = pd.DataFrame(columns = column_names,index = list(range(len(filtered_df))))
for i in range(len(filtered_df)):
    df_md_1.iloc[i,0] = filtered_df.iloc[i,1]
    df_md_1.iloc[i,1] = filtered_df.iloc[i,10]
    if len(str(df.iloc[i,8]))<1:
        df_md_1.iloc[i,2] = f"['{filtered_df.iloc[i,7]}']"
    elif len(str(df.iloc[i,9]))<1:
        df_md_1.iloc[i,2] = f"['{filtered_df.iloc[i,7]}','{filtered_df.iloc[i,8]}']"
    else:
        df_md_1.iloc[i,2] = f"['{filtered_df.iloc[i,7]}','{filtered_df.iloc[i,8]}','{filtered_df.iloc[i,9]}']"
print(df_md_1[0:])


# In[86]:


# save dataframe to csv file
df_md_1.to_csv('test_text_from_model2.csv',index=False)


# In[ ]:




