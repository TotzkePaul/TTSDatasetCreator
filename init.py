import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


workspace = input("Name of Project: ")

workspace_dir = "../Workspace"

mkdir('{}/{}'.format(workspace_dir, workspace))
mkdir('{}/{}/wavs'.format(workspace_dir, workspace))
mkdir('{}/{}/transcript'.format(workspace_dir, workspace))
mkdir('{}/{}/gecko'.format(workspace_dir, workspace))
mkdir('{}/{}/src'.format(workspace_dir, workspace))

mkdir('../Models')
mkdir('../Datasets')
mkdir('../DeepSpeechModels')