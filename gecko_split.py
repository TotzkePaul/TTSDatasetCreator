import argparse
import glob
import os
from pydub import AudioSegment
import random
from pydub.generators import WhiteNoise
import json


def partition(l, pred):
    yes, no = [], []
    for e in l:
        if pred(e):
            yes.append(e.rstrip('\n\r'))
        else:
            no.append(e.rstrip('\n\r'))
    return yes, no


def terms_to_string(terms):
    sentence = ' '.join(term["text"] for term in terms)
    return sentence.replace(' ,', ',').replace(' .', '.')


def main():
    parser = argparse.ArgumentParser(description='Split audio based on gecko transcript.')
    parser.add_argument('--output', default='../Datasets', help='output directory')
    parser.add_argument('--media', default='../Workspace', help='input directory')
    parser.add_argument('--name', required=True, help='name of project')
    parser.add_argument('--speaker', help='only include this speaker')
    parser.add_argument('--min', default=750, type=int, help='min duration of clip')
    parser.add_argument('--max', default=13000, type=int, help='max duration of clip')
    parser.add_argument('--lead_silence', default=0, help='amount to trim at beginning')
    parser.add_argument('--trail_silence', default=0, help='amount to trim at end')
    parser.add_argument('--partition', default=.9, help='max duration of clip')
    args = parser.parse_args()

    output_dir = '{0}/{1}'.format(args.output, args.name)
    workspace_dir = '{0}/{1}'.format(args.media, args.name)
    audio_dir = '{0}/wavs/22050/'.format(workspace_dir)
    transcript_dir = '{0}/transcripts'.format(workspace_dir)
    gecko_dir = '{0}/gecko'.format(workspace_dir)

    print(output_dir, audio_dir, transcript_dir, gecko_dir)

    lead_silence = AudioSegment.silent(duration=args.lead_silence)
    trail_silence = AudioSegment.silent(duration=args.trail_silence)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(output_dir+'/wavs'):
        os.mkdir(output_dir+'/wavs')

    pad_silence = (args.lead_silence + args.trail_silence) > 0
    train_txt = []
    dummy_txt = []

    train_set = []
    val_set = []
    wg_set = []

    flowtron_txt = []

    cntr = 0

    for filepath in sorted(glob.glob("{0}/*.transcript.json".format(transcript_dir)), key=os.path.getmtime):
        print(filepath)
        transcript_basename = os.path.basename(filepath)
        gecko_path = '{0}/{1}'.format(gecko_dir, transcript_basename)
        audio_basename = os.path.splitext(os.path.splitext(transcript_basename)[0])[0]
        audio_path = '{0}/{1}'.format(audio_dir, audio_basename)
        audio_file = AudioSegment.from_wav(audio_path)

        base_clip_name = os.path.splitext(os.path.basename(audio_path))[0]

        is_auto_transcript = True

        if os.path.exists(gecko_path):
            print('gecko transcript exits', gecko_path)
            filepath = gecko_path
            is_auto_transcript = False

        with open(filepath) as json_file:
            json_data = json.load(json_file)
            print(filepath)
            for i, monologue in enumerate(json_data["monologues"]):

                sentence = terms_to_string(monologue["terms"])

                clip_name = "{0}_{1:0>4d}.wav".format(base_clip_name, i)

                clip_path = "{0}/wavs/{1}".format(output_dir, clip_name)

                start = monologue["start"] * 1000
                end = monologue["end"] * 1000

                duration = end - start

                is_empty = len(sentence) == 0 or sentence.isspace()

                include = args.min <= duration <= args.max and not is_empty

                if args.speaker is not None and len(args.speaker) > 0 and monologue["speaker"]["id"] != args.speaker:
                    include = False

                if duration > 9000:
                    cntr = cntr + 1
                    print("long clips: ", cntr)

                print('path:', clip_path, 'Sentence:', sentence, 'duration:', duration, "include:", include)

                if not include:
                    continue
                if sentence[-1] != '.' or sentence[-1] != ',' or sentence[-1] != '?':
                    sentence = sentence + '.'
                if pad_silence:
                    clip_audio = lead_silence + audio_file[start:end] + trail_silence
                else:
                    clip_audio = audio_file[start:end]

                clip_audio.export(clip_path, format="wav")

                wg_set.append('{}/wavs/{}'.format(output_dir, clip_name, sentence))
                train_txt.append('{}/wavs/{}|{}'.format(output_dir, clip_name, sentence))
                dummy_txt.append('{}/{}|{}'.format('DUMMY', clip_name, sentence))
                flowtron_txt.append('{}/wavs/{}|{}|0'.format(output_dir, clip_name, sentence))

    dummy_filelist = '{}/dummy_{}_all.txt'.format(output_dir, args.name)
    complete_filelist = '{}/{}_all.txt'.format(output_dir, args.name)
    train_filelist = '{}/{}_train.txt'.format(output_dir, args.name)
    validate_filelist = '{}/{}_val.txt'.format(output_dir, args.name)
    wg_filelist = '{}/{}_wg.txt'.format(output_dir, args.name)

    with open(complete_filelist, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_txt))
    with open(dummy_filelist, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dummy_txt))

    lines = train_txt  # open(complete_filelist, 'r', encoding='utf-8').readlines()
    lines1, lines2 = partition(lines, lambda x: random.random() < args.partition)

    train_set.extend(lines1)
    val_set.extend(lines2)

    flowtron_train, flowtron_val = partition(flowtron_txt, lambda x: random.random() < args.partition)

    flowtron_val_file = '{}/{}_flowtron_val.txt'.format(output_dir, args.name)
    flowtron_train_file = '{}/{}_flowtron_train.txt'.format(output_dir, args.name)

    with open(wg_filelist, 'w', encoding='utf-8') as f:
        f.write('\n'.join(wg_set))

    with open(flowtron_train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(flowtron_train))
    with open(flowtron_val_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(flowtron_val))

    with open(train_filelist, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_set))
    with open(validate_filelist, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(val_set))


if __name__ == "__main__":
    main()


