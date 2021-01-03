git clone https://github.com/TotzkePaul/TTSDatasetCreator/

mkdir ./Models

mkdir ./Workspace

mkdir ./Datasets

mkdir ./DeepSpeechModels

Go to https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2
and deepspeech-0.8.2-models.tflite, deepspeech-0.8.2-models.pbmm and deepspeech-0.8.2-models.scorer to DeepSpeechModels

cd ./TTSDatasetCreator

conda env create -f ./win10_conda_env/transcribe.yml

spilt.py --name=Offerman

transcribe.py --model ../DeepSpeechModels/deepspeech-0.8.2-models.pbmm --scorer ../DeepSpeechModels/deepspeech-0.8.2-models.scorer --name=Offerman

gecko_split.py  --output ../Datasets --media ../Media/ --name Offerman --min 0 --max 20000