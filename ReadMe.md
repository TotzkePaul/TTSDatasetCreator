git clone https://github.com/TotzkePaul/TTSDatasetCreator/

mkdir ./Models

mkdir ./Workspace

mkdir ./Datasets

mkdir ./DeepSpeechModels

Go to https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2
and deepspeech-0.8.2-models.pbmm and deepspeech-0.8.2-models.scorer to DeepSpeechModels

cd ./TTSDatasetCreator

`conda env create -f ./win10_conda_env/transcribe.yml`

Pick a Workspace name (eg Offerman) and save source audio to ../Workspace/{WorkspaceName}/src 

`python spilt.py --name={WorkspaceName}`

`python transcribe.py --model ../DeepSpeechModels/deepspeech-0.8.2-models.pbmm --scorer ../DeepSpeechModels/deepspeech-0.8.2-models.scorer --name={WorkspaceName}`


Review corresponding wav and transcript files using https://gong-io.github.io/gecko/. 

Save updated transcript to ../Workspace/{WorkspaceName}/gecko with the same transcript file

`python gecko_split.py  --output ../Datasets --media ../Media/ --name {WorkspaceName} --min 0 --max 12000`

The completed dataset now exists in ./Datasets/{WorkspaceName}

{WorkspaceName}_wg.txt is for waveglow; 

{WorkspaceName}_train.txt and {WorkspaceName}_val.txt are for tacotron2

{WorkspaceName}_flowtron_train.txt and {WorkspaceName}_flowtron_val.txt are for flowtron and mellowtron