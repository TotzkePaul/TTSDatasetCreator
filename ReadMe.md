# Prereqs

FFmpeg, cuda, cudnn

Miniconda3 Windows 64-bit (Python 3.8):

https://docs.conda.io/en/latest/miniconda.html

install pytorch

`conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge`

ffmpeg for Win10: 
https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2021-07-21-12-38

ffmpeg-N-103022-gf614390ecc-win64-gpl.zip

# Overview

1. init.py creates the Workspace folder structure.

   |---Datasets
   |---DeepSpeechModels
   |---Models
   |---TTSDatasetCreator
   |-----win10_conda_env
   |---Workspace
   |-----$(workspace)
   |-------gecko
   |-------speakers
   |-------src
   |-------transcripts
   |-------wavs

2. split.py converts the audio files in ./Workspace/$(workspace)/src into TacoTron-compatible 16000hz & 22050hz wavs in the wavs folder with --length X, were X is the max minutes per part ( eg --length 60 for a 2 hour audio file will make 2 parts)
3. (optional) segment.py creates $(wav_file).segment.json in the transcripts folder that marks where speech is taking place
4. transcribe.py uses Mozilla's Deepspeech to do Speech To Text and creates Gecko compatible $(wav_file).transcript.json which uses $(wav_file).segment.json if it exists or does it dynamically
5. Open $(wav_file).transcript.json in Gecko https://gong-io.github.io/gecko/ and save results to the gecko folder
6. gecko_split.py creates the finished dataset

Colab Example: 
https://colab.research.google.com/drive/1r-tsd5XyqAWtnhAVSqHFUKdPa_ViAUVU?usp=sharing

# Init

`git clone https://github.com/TotzkePaul/TTSDatasetCreator/`

`cd ./TTSDatasetCreator`

Tested with python=3.7

Create Workspace, Models, DeepSpeechModels and Dataset Folders

`python ./TTSDatasetCreator/init.py`

Go to https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2
and deepspeech-0.8.2-models.pbmm and deepspeech-0.8.2-models.scorer to DeepSpeechModels

Win10: 

`curl https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-models.pbmm -o '../DeepSpeechModels/deepspeech-0.8.2-models.pbmm'`

`curl https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-models.scorer -o '../DeepSpeechModels/deepspeech-0.8.2-models.scorer'`

Linux: 

`!wget -O '../DeepSpeechModels/deepspeech-0.8.2-models.pbmm' https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-models.pbmm`

`!wget -O '../DeepSpeechModels/deepspeech-0.8.2-models.scorer' https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-models.scorer`

# Save mp3, wav and m4a files ../Workspace/{$Name}/src 

Pick a Workspace name (eg Rogan) and save source audio to `../Workspace/{WorkspaceName}/src`

# Split long in to manageable chunks

split.py uses pydub to split files into 15 minute wav files. 16000hz for deepspeech and 22050 for Tacotron

`python split.py --name={WorkspaceName}`

# Auto Speaker Labeling for Multi-Speaker audio (optional) 

For multiple speakers, go to https://gong-io.github.io/gecko/ and open a wav file from `../Workspace/$Name/wavs`

Segment 10 5-second clips for each speaker and label the speaker (Left hand side - Segment Labeling). 

Hit save and save to `../Workspace/$Name/speakers` as `same_filename.wav.speaker.json`

`python speaker_split.py --output speakers --name Rogan --min 3000 --max 8000`

This will create subfolders under `../Workspace/{$Name}/speakers` with the names of the speaker labels.

Those folders contain 3-second to 8-second clips of the speaker. These will be used to auto-tag other files.

The more clips, the longer the next step runs but (ideally) more accurate segmentation labeling.

# Segment

`python segment.py --name $Name`

This will create `same_filename.wav.segment.json` for each wav in `../Workspace/$Name/wavs`

You can review and edit this file in [gecko](https://gong-io.github.io/gecko/)

# Transcribe

`python transcribe.py --model ../DeepSpeechModels/deepspeech-0.8.2-models.pbmm --scorer ../DeepSpeechModels/deepspeech-0.8.2-models.scorer --name={WorkspaceName}`

This will create `same_filename.wav.transcript.json` for each wav in `../Workspace/$Name/wavs`

Have approximate transcripts? Save txt files to `../Workspace/$Name/hot_words` as audiofile1.wav.txt or audiofile1.txt 

Currently, this just extracts all unique words and boosts them by the same amount.

If you see warnings like missing dll, either update tensorflow or use the following:

`conda install -c anaconda cudatoolkit=10.1`

`conda install -c anaconda cudnn`

# Gecko_Split

Review corresponding wav and transcript files using [gecko.](https://gong-io.github.io/gecko/)

To override the auto generated transcript, save the gecko save file as `same_filename.wav.transcript.json` in `../Workspace/$Name/gecko`

`python gecko_split.py  --output ../Datasets --name {$Name} --min 0 --max 12000 --speaker "Joe Rogan"` 

`--speaker` is optional. It filters the clips where gecko's speaker label is the name.

This creates a subfolder in `../Datasets/` and creates a tacotron, waveglow and flowtron compatibile dataset.

The complete dataset for clips 0 to 12 seconds now exists in ./Datasets/{WorkspaceName}

{WorkspaceName}_wg.txt is for waveglow; 

{WorkspaceName}_train.txt and {WorkspaceName}_val.txt are for tacotron2

{WorkspaceName}_flowtron_train.txt and {WorkspaceName}_flowtron_val.txt are for flowtron and mellowtron