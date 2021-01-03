import glob
import argparse
import os
from pydub import AudioSegment
from pydub.utils import make_chunks


def split_single(output, name, basefilename, sound_file, frame_rate, length):
    audio_data = sound_file.set_sample_width(2).set_channels(1).set_frame_rate(frame_rate)
    chunk_length_ms = length * 60 * 1000  # pydub calculates in millisec
    if chunk_length_ms == 0 or audio_data.duration_seconds * 1000 < chunk_length_ms:
        chunks = [audio_data]
    else:
        chunks = make_chunks(audio_data, chunk_length_ms)  # Make chunks of length minutes

    output_dir = "{0}/{1}".format(output, frame_rate)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i, chunk in enumerate(chunks):
        chunk_name = "{0}/{1}_part{2:0>3d}.wav".format(output_dir, basefilename, i)
        print(chunk_name)
        chunk.export(chunk_name, format="wav")


def main():
    parser = argparse.ArgumentParser(description='Split audio in fixed lengths.')
    parser.add_argument('--output', default='wavs', help='output directory')
    parser.add_argument('--input', default='../Workspace', help='input directory')
    parser.add_argument('--name', required=True, help='name of project')
    parser.add_argument('--length', default=15, help='length in minutes of parts')
    args = parser.parse_args()

    path = '{0}/{1}/src'.format(args.input, args.name)
    output_path = '{0}/{1}/{2}'.format(args.input, args.name, args.output)

    # Export all of the individual chunks as wav files
    if os.path.isdir(path):
        print('directory:', path)
        for filepath in glob.glob('{0}/*.wav'.format(path)):
            basefilename = os.path.splitext(os.path.basename(filepath))[0]
            print('basefilename:', basefilename, 'path:', filepath)
            sound_file = AudioSegment.from_file(filepath)
            split_single(output_path, args.name, basefilename, sound_file, 16000, args.length)
            split_single(output_path, args.name, basefilename, sound_file, 22050, args.length)

        for filepath in glob.glob('{0}/*.mp3'.format(path)):
            basefilename = os.path.splitext(os.path.basename(filepath))[0]
            print('basefilename:', basefilename, 'path:', filepath)
            sound_file = AudioSegment.from_file(filepath)
            split_single(output_path, args.name, basefilename, sound_file, 16000, args.length)
            split_single(output_path, args.name, basefilename, sound_file, 22050, args.length)

    elif os.path.isfile(path):
        print('path:', path)
        sound_file = AudioSegment.from_file(path)
        split_single(output_path, args.name, args.name, sound_file, 16000, args.length)
        split_single(output_path, args.name, args.name, sound_file, 22050, args.length)
    else:
        print("Not a file or directory")


if __name__ == "__main__":
    main()
