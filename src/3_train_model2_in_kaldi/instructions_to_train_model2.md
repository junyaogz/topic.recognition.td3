# [Author]: Jun Yao
# [UNI]: jy2992
# [Date]: 2021-11-28
# [Email]: jun.yao@columbia.edu

# [Description] 
The purpose of this phase is to train model 1 in kaldi. The whole process runs for 5 days on the author's machine.  

Instructions here largely follow the scripts provided by egs, the Professor and TAs,  
I only modified two files:  
(1) `cmd.sh` to let the script run localy,   
(2) `run.sh`, I commented out steps 16-19 since I only need tri3 model.

Although an all-in-one .sh file is fantastic, strange problems occur when runing these scripts, so I break down this phase into 6 steps and execute
them one by one.

Open terminal in the folder where kaldi-trunk will sit, such as the homedir `~` in linux.
All folders in the script are relative to this root folder, in each step I return to the root folder to facilitate following steps. Please copy and paste to run the following commands in each step.

Input of this phase:  
official TED-LIUM v3 dataset: http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz  
author's modified version of the egs script: https://github.com/junyaogz/kaldi.git  

Output of this script:  
decoded text that will be used in later phase: kaldi-trunk/egs/tedlium/s5_r3/exp/tri3/decode_test/log


# step 1 - install build tools
```bash
# Run the following commands under folder `~`, line by line:
#!/bin/bash
sudo apt-get -y install build-essential
sudo apt-get -y install git
sudo apt-get -y update
sudo apt-get -y install linux-headers-$(uname -r)
sudo apt-get -y install flac libflac-dev; 
# this will take a while
sudo apt-get -y install libatlas*; 
sudo apt-get -y install subversion; 
sudo apt-get -y install speex libspeex-dev; 
sudo apt-get -y install python-numpy swig; 
# if not-found libgstreamer-1.0-dev, just skip and continue
sudo apt-get -y install gstreamer-1.0 libgstreamer-1.0-dev; 
# this will take a while
sudo apt-get -y install libgstreamer-plugins*; 
# install python-pip on ubuntu 18.04 if pip is missing
sudo apt-get -y install python-pip; pip install --upgrade pip; pip install ws4py; pip install tornado==4; 
sudo apt-get -y install python-anyjson; 
sudo apt-get -y install libyaml-dev; pip install pyyaml; 
sudo apt-get -y install libjansson-dev;
sudo apt-get -y install gnome-applets
sudo apt-get -y install sox
sudo apt-get -y install unzip
# the author's code sits on github
git clone https://github.com/junyaogz/kaldi.git kaldi-trunk
```

# STEP 2 - install CUDA and kaldi dependencies
```bash
# Install Cuda
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
chmod +x cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
# Check installation using the following command:
nvidia-smi
#Mon Nov 22 15:55:13 2021       
#+-----------------------------------------------------------------------------+
#| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
#|-------------------------------+----------------------+----------------------+
#| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#|===============================+======================+======================|
#|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
#| N/A   40C    P0    67W / 149W |      0MiB / 11441MiB |    100%      Default |
#+-------------------------------+----------------------+----------------------+
#                                                                               
#+-----------------------------------------------------------------------------+
#| Processes:                                                       GPU Memory |
#|  GPU       PID   Type   Process name                             Usage      |
#|=============================================================================|
#|  No running processes found                                                 |
#+-----------------------------------------------------------------------------+
# check dependencies, install missing dependencies, 
# if any, like `sudo apt-get install gfortran`, `sudo ./extras/install_mkl.sh`
cd ./kaldi-trunk/tools/
extras/check_dependencies.sh
sudo apt-get install gfortran
sudo ./extras/install_mkl.sh
cd ../..
```

# STEP 3 - compile kaldi
```bash
# Run following commands to compile kaldi
cd ./kaldi-trunk/tools/
make

cd ../src
./configure 
# this will take a while
make
cd ..
```

# STEP 4 - run egs/YesNo to verify the build
```bash
# Run an example `egs/YesNo` to verify the output of the above builds
cd ./kaldi-trunk/egs/yesno/s5
./run.sh > run.log
# expected output:
# %WER 0.00 [ 0 / 232, 0 ins, 0 del, 0 sub ] exp/mono0a/decode_test_yesno/wer_10_0.0
cd ../..
cd ../..
```

# STEP 5 - run egs/tedlium/s5_s3
```bash
# dowloading data will take a long time, 
# download it from `http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz` in advance
# palce the file under folder `~/kaldi-trunk/egs/tedlium/s5_r3/db/` and unzip it with command `tar xf "TEDLIUM_release-3.tgz"`
# check configuration in `egs/tedlium/s5_r3/cmd.sh` to make sure that all scripts are run locally
# run `egs/tedlium/s5_r3/run.sh` to get a base model
cd ./kaldi-trunk/egs/tedlium/s5_r3/
# it will take very long time to train the models locally
# you can use `nohup [command]` to keep it running in the backgroud when disconnected from the vm
nohup ./run.sh > run.log
cd ../..
cd ../..
# if execution is interrupted, modify the `stage` variable in run.sh and run again.
```

# STEP 6 - copy decode log for future use 
```bash
# copy tri3 model logs to phase 4 folder
cp -R ./kaldi-trunk/egs/tedlium/s5_r3/exp/tri3/decode_test/log ../4_transform_decoded_text_from_model2/log
```

