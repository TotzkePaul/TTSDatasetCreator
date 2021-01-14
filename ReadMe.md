# Prereqs

FFmpeg, cuda, cudnn


# Init

`git clone https://github.com/TotzkePaul/TTSDatasetCreator/`

`cd ./TTSDatasetCreator`

`conda env create -f ./win10_conda_env/transcribe.yml`

Create Workspace, Models, DeepSpeechModels and Dataset Folders

`python ./TTSDatasetCreator/init.py`

Go to https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2
and deepspeech-0.8.2-models.pbmm and deepspeech-0.8.2-models.scorer to DeepSpeechModels

# Save mp4, wav and m4a files ../Workspace/{$Name}/src 

Pick a Workspace name (eg Rogan) and save source audio to `../Workspace/{WorkspaceName}/src`

# Split

split.py uses pydub to split files into 15 minute wav files. 16000hz for deepspeech and 22050 for Tacotron

`python split.py --name={WorkspaceName}`

# Segment

For multiple speakers, go to https://gong-io.github.io/gecko/ and open a wav file from `../Workspace/$Name/wavs`

Segment 10 5-second clips for each speaker and label the speaker (Left hand side - Segment Labeling). 

Hit save and save to `../Workspace/$Name/speakers` as `same_filename.wav.speaker.json`

`python speaker_split.py --output speakers --name Rogan --min 3000 --max 8000`

This will create subfolders under `../Workspace/{$Name}/speakers` with the names of the speaker labels.

Those folders contain 3-second to 8-second clips of the speaker. These will be used to auto-tag other files.

The more clips, the longer the next step runs but (ideally) more accurate segmentation. 

`python segment.py --name $Name`

This will create `same_filename.wav.segment.json` for each wav in `../Workspace/$Name/wavs`

You can review and edit this file in [gecko](https://gong-io.github.io/gecko/)

# Transcribe

`python transcribe.py --model ../DeepSpeechModels/deepspeech-0.8.2-models.pbmm --scorer ../DeepSpeechModels/deepspeech-0.8.2-models.scorer --name={WorkspaceName}`

This will create `same_filename.wav.transcript.json` for each wav in `../Workspace/$Name/wavs`

# Gecko_Split

Review corresponding wav and transcript files using [gecko.](https://gong-io.github.io/gecko/)

To override the auto generated transcript, save the gecko save file as `same_filename.wav.transcript.json` in `../Workspace/$Name/gecko`

`python gecko_split.py  --output ../Datasets --media ../Media/ --name {$Name} --min 0 --max 12000`

This creates a subfolder in `../Datasets/` and creates a tacotron, waveglow and flowtron compatibile dataset.

The complete dataset for clips 0 to 12 seconds now exists in ./Datasets/{WorkspaceName}

{WorkspaceName}_wg.txt is for waveglow; 

{WorkspaceName}_train.txt and {WorkspaceName}_val.txt are for tacotron2

{WorkspaceName}_flowtron_train.txt and {WorkspaceName}_flowtron_val.txt are for flowtron and mellowtron