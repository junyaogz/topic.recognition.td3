#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# [Author]: Jun Yao
# [Date]: 2021-12-03

# [Description] 
# this file prepares the golden labels for topic recognition for TEDLIUM3 dataset,
# it has the following functionalities
# (1) query all links of the 2351 videos from ted.com. The year of some videos 
# are not matched with the year posted, links of these videos have to be fixed manually
# (2) using the links of the videos, send http request to ted.com and get extra metadata,
# such as author, upload date, duration, title, topic labels.
# (3) the final verified golden labels file tedlium3_topic_labels.csv is published
# via Github under MIT license at https://github.com/junyaogz/topic.labels

# input of this script:
# ./stm/*

# output of this script:
# tedlium3_topic_labels.csv

# [Instructions] 
# WARNING! this file runs for more than 36 hours! 
# it needs human intervention to fix some errors since there are some errors in the year 
# of the videos and the search engine of ted.com cannot provide correct links for all videos.
# frequent requests are prohibited by ted.com, so timeout will occur, the last part of this
# code has to be executed repeatedly until all videos are processed succesfully.


# In[2]:


import re
import requests
from bs4 import BeautifulSoup
import random
from pathlib import Path
import time

root_path = "/Volumes/SanDisk512/TEDLIUM_release-3/data/stm/"
ted_talk_root = "http://www.ted.com/talks"
ted_web_root = "http://www.ted.com"

def query_video_info(file_name):
    q_sentence = ""

    # find a sentence that has at least 6 words
    # split the sentence, remove date time, <NA> and <UNK>
    with open(root_path+file_name) as file:
        videos = []
        for line in file:
            sleep_time = random.randint(1,4)
            print(f"sleep {sleep_time} seconds to avoid blocking")
            time.sleep(sleep_time)
            tmp = line.split("<NA>")[1].replace("<unk>","").replace("'","").strip()
            #print(tmp)
            q_sentence = tmp
            if len(q_sentence.split(" ")) < 6:
                continue
            
            #remove non-ascii characters
            q_sentence = q_sentence.encode("ascii", "ignore").decode()

            # join words with "+"
            url_q_sentence = "+".join(q_sentence.split(" "))
            #print("Query sentence for {0} is: {1} ".format(file_name, url_q_sentence))

            # add prefix https://www.ted.com/talks?sort=relevance&q=
            request_url = ted_talk_root+"?sort=relevance&q=" + url_q_sentence
            print(request_url)

            # send page request, get the response page
            res1 = ""
            try:
                res1 = requests.get(request_url, timeout=5)
                #print(res1.status_code)
            except:
                print("error: request timeout")
                continue

            res_1_text = ""
            res_1_text = res1.text
            #with open(rootpath + "res1.txt") as file:
            #    res_1_text = file.readlines()

            soup = BeautifulSoup(str(res_1_text))
            #print(soup.prettify())

            candidate_videos = []
            divs = soup.find_all('div', {"class","media__message"})
            for div in divs:
                #print(div.text)   
                link = div.find('a', {"class","ga-link"})
                link_url = link["href"].strip()
                year = div.find('span', {"class","meta__val"})
                tmp = re.findall('[0-9]+', year.text)
                #print(tmp[0])  
                if link_url.startswith("/talks/"):
                    candidate_videos.append([link_url,tmp[0]])

            #print(candidate_videos)

            years_to_match = re.findall('[0-9]+', file_name)
            year_to_match = 0
            if(len(years_to_match)>1):
                year_to_match = years_to_match[1]
            else:
                year_to_match = years_to_match[0]
            #print(year_to_match)
            
            videos = []
            # from the list of search results, match the author and year(maybe +10), select that item
            for v in candidate_videos:
                if abs(int(year_to_match)-int(v[1])) < 3:
                    videos.append(v)

            print(f"found {len(videos)} videos: {videos}")
            #assert len(videos)>=1, "expected matched video num is no less than 1"
            if len(videos)==1:
                break
                
    if len(videos)==0:
        return 0
    
    return videos[0]


# for each stm file, do the following steps
# read stm file

video_uris = []
count = 0
for p in Path(root_path).glob('*.stm'):
    count = count + 1
    fname = p.name
    print(f"------<{count}>----->filepath:{root_path}{fname}")
    videos = query_video_info(fname)
    if videos == 0:
        print("error: no video is found.")
        video_uris.append([fname,"not found"])
    else:
        print(f"selected: {videos[0]}")
        video_uris.append([fname,videos[0]])
    
    sleep_time = random.randint(1,5)
    print(f"sleep {sleep_time} seconds to avoid blocking")
    time.sleep(sleep_time)


# In[3]:


print(f"len of video uris is {len(video_uris)}")
#print(video_uris)

def saveListToFile(list_name, path_to_save):
    fp = open(path_to_save,"w") 
    num = 0
    for line in list_name:
        num = num + 1
        fp.writelines(f"{num}, {line[0]}, {line[1]}\n")    
    fp.close() 

saveListToFile(video_uris, "video_uris.txt")

# remark:
# the above script runs for about 36 hours and outputs file  video_uris.txt
# it contains some missing values (78 out of 2351 videos can not be matched with the videos from ted.com
# using transcript and year when I run it), 
# these videos have to be searched and labelled manually.
# I fixed the missing values manually, added column names, created a new file video_uris_repaired.csv


# In[16]:


import pandas as pd

url_list_file = "video_uris_repaired.csv"

# send page request, get the response page
def extract_video_properties(video_url):
    request_url = ted_web_root + video_url
    print(request_url)

    # send page request, get the response page
    res2 = ""
    try:
        res2 = requests.get(request_url, timeout=5)
        #print(res2.status_code)
    except:
        print("error: request timeout")
        return None
    
    res_2_text = ""
    res_2_text = res2.text
    #with open(rootpath + "res2.txt") as file:
    #    res_2_text = file.readlines()

    soup = BeautifulSoup(str(res_2_text))
    #print(soup.prettify())
    
    #get video intro
    metas = soup.find_all('meta',{"property":"og:video:tag"})
    #choose the first three tags as topics
    topics = ["","",""]
    if len(metas)>0:
        topics[0] = metas[0]["content"].strip()        
    if len(metas)>1:
        topics[1] = metas[1]["content"].strip()
    if len(metas)>2:
        topics[2] = metas[2]["content"].strip()
    #print(topics)
    
    authors = soup.find_all('meta',{"name":"author"})
    author=""
    if len(authors)>0:
        author = authors[0]["content"].strip()
    #print(author)
    
    titles = soup.find_all('meta',{"property":"og:title"})
    title=""
    if len(titles)>0:
        title = titles[0]["content"].strip()
    #print(title)
    
    descs = soup.find_all('meta',{"property":"og:description"})
    desc=""
    if len(descs)>0:
        desc = descs[0]["content"].strip()
    #print(desc)
    
    uploaddates = soup.find_all('meta',{"itemprop":"uploadDate"})
    uploaddate=""
    if len(uploaddates)>0:
        uploaddate = uploaddates[0]["content"].strip()[0:10]
    #print(uploaddate)
    
    durations = soup.find_all('meta',{"itemprop":"duration"})
    duration=""
    if len(durations)>0:
        duration = durations[0]["content"].strip()[2:]
    #print(duration)
    return [title,author,uploaddate, request_url, duration, topics[0],topics[1],topics[2]]

# read all urls from file
v_data = pd.read_csv(url_list_file)
v_data.head()


# In[30]:


df = pd.DataFrame(v_data)
# add new columns to the dataframe
df = df.assign(TITLE="")
df = df.assign(AUTHOR="")
df = df.assign(UPLOAD_DATE="")
df = df.assign(DURATION="")
df = df.assign(TOPIC1="")
df = df.assign(TOPIC2="")
df = df.assign(TOPIC3="")

df_len = len(df)
print(df_len)

df.head()


# In[55]:


# run this part of code repeatedly until all videos are extracted correctly.
for i in range(df_len):
    if df.iloc[i,3] == "": #title is empty
        video_props = extract_video_properties(df.iloc[i,2].strip())
        print(f"video {i+1} properties: {video_props}")
        if video_props:
            df.iloc[i,3] = video_props[0] #title
            df.iloc[i,4] = video_props[1] #author
            df.iloc[i,5] = video_props[2] #upload_date
            df.iloc[i,6] = video_props[4] #duration
            df.iloc[i,7] = video_props[5] #topic1
            df.iloc[i,8] = video_props[6] #topic2
            df.iloc[i,9] = video_props[7] #topic3
        else:
            print("sleep 3 seconds to avoid blocking")
            time.sleep(3)
            continue
        
filtered_df = df[df['TITLE'] == ""]
if len(filtered_df)==0:
    print("successful")
else:
    print(f"{len(filtered_df)} videos failed, please run this part of code again.")

df.head()

#print(df.iloc[1272,2])


# In[57]:


# if video 1272 can not be retrieved, fix it manually using the following code
df.iloc[1272,3] = "The unstoppable walk to political reform" #title
df.iloc[1272,4] = "Lawrence Lessig" #author
df.iloc[1272,5] = "2014-04-04" #upload_date
df.iloc[1272,6] = "13M44S" #duration
df.iloc[1272,7] = "corruption" #topic1
df.iloc[1272,8] = "democracy" #topic2
df.iloc[1272,9] = "government" #topic3

df.head()


# In[62]:


# save the pd to csv
for i in range(df_len):
    df.iloc[i,1] = df.iloc[i,1].strip() #remove spaces
    df.iloc[i,2] = df.iloc[i,2].strip() #remove spaces
    
df.to_csv('tedlium3_topic_labels.csv',index=False)


# In[ ]:




