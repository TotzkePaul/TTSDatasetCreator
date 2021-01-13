import argparse
import glob
import os
from pydub import AudioSegment
import random
from pydub.generators import WhiteNoise
import json


def main():
    parser = argparse.ArgumentParser(description='Split audio based on gecko transcript.')
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('--media', default='../Workspace', help='input directory')

    parser.add_argument('--name', required=True, help='name of project')
    parser.add_argument('--min', default=3000, type=int, help='min duration of clip')
    parser.add_argument('--max', default=8000, type=int, help='max duration of clip')
    args = parser.parse_args()

    speaker_dir = '{0}/{1}/{2}'.format(args.media, args.name, args.output)
    if not os.path.exists(speaker_dir):
        os.makedirs(speaker_dir)

    audio_dir = '{0}/{1}/wavs/16000/'.format(args.media, args.name)

    for filepath in sorted(glob.glob("{0}/*.speaker.json".format(speaker_dir)), key=os.path.getmtime):
        print(filepath)
        transcript_basename = os.path.basename(filepath)
        audio_basename = os.path.splitext(os.path.splitext(transcript_basename)[0])[0]
        audio_path = '{0}/{1}'.format(audio_dir, audio_basename)
        audio_file = AudioSegment.from_wav(audio_path)

        base_clip_name = os.path.splitext(os.path.basename(audio_path))[0]

        with open(filepath) as json_file:
            json_data = json.load(json_file)
            print(filepath)
            for i, monologue in enumerate(json_data["monologues"]):
                start = monologue["start"] * 1000
                end = monologue["end"] * 1000

                duration = end - start

                if not args.min < duration < args.max:
                    continue

                if monologue["speaker"]["id"] is None or monologue["speaker"]["id"] == '<NA>':
                    continue

                clip_name = "{0}_{1:0>4d}.wav".format(base_clip_name, i)

                clip_path = "{0}/{1}/{2}".format(speaker_dir, monologue["speaker"]["id"], clip_name)

                if not os.path.isdir(os.path.dirname(clip_path)):
                    os.mkdir(os.path.dirname(clip_path))



                print('path:', clip_path, 'duration:', duration)

                clip_audio = audio_file[start:end]

                clip_audio.export(clip_path, format="wav")


if __name__ == "__main__":
    main()


