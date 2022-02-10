#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# [Author]: Jun Yao
# [Date]: 2021-12-03

# [Description] 
# this file prepares the train and test dataset for model 1

# [Instructions] 
# official transcript files (2351 .stm files) are copied to the folder where this script sits
# to make this folder all-inclusive, so you can put this folder anywhere you want.

# input of this script:
# ./stm/*
# tedlium3_topic_labels.csv

# output of this script:
# stm_transcript_labels.csv


# In[21]:


import pandas as pd

stm_path = "./stm/"

# read gold labels file
df = pd.read_csv("tedlium3_topic_labels.csv")

# add a new column to the dataframe
df = df.assign(TRANSCRIPT="")

print(len(df))
df.head()


# In[40]:


def read_stm_transcript(file_name):
    #print(file_name)
    full_path = stm_path + file_name
    #print(f"file path: {full_path}")
    trans = []
    with open(full_path) as file:
        for line in file:
            trans.append(line.split("<NA>")[1].replace("<unk>","").replace("'","").replace("\n","").strip())
    return ". ".join(trans)

# read all transcripts
df_len = len(df)
for i in range(df_len):
    trans = read_stm_transcript(df.iloc[i,1]) #notice the index of the column FILENAME
    df.iloc[i,10] = trans
    if(i%100==0):
        print(f"========>processed {i} files")
print("success")

df.head()


# In[55]:


# prepare the data for model 1
column_names = ["titles", "summaries", "terms"]
df_md_1 = pd.DataFrame(columns = column_names,index = list(range(df_len)))
for i in range(df_len):
    df_md_1.iloc[i,0] = df.iloc[i,1]
    df_md_1.iloc[i,1] = df.iloc[i,10]
    if len(str(df.iloc[i,8]))<1:
        df_md_1.iloc[i,2] = f"['{df.iloc[i,7]}']"
    elif len(str(df.iloc[i,9]))<1:
        df_md_1.iloc[i,2] = f"['{df.iloc[i,7]}','{df.iloc[i,8]}']"
    else:
        df_md_1.iloc[i,2] = f"['{df.iloc[i,7]}','{df.iloc[i,8]}','{df.iloc[i,9]}']"
df_md_1.head()


# In[56]:


# save dataframe to csv file
df_md_1.to_csv('stm_transcript_labels.csv',index=False)


# In[47]:





# In[ ]:




