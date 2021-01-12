from __future__ import absolute_import, division, print_function
import argparse
import sys
import json
import time
import glob
import os
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, detect_silence
from pydub.utils import make_chunks


def remove_inhales(segments, amplitudes, threshold, min_duration, max_duration):
    output_segments = []

    for start, stop in segments:
        candidate = amplitudes[start:stop]

        quartiles = np.percentile(candidate, [10, 25, 50])

        front = remove_initial_inhales(start, stop, candidate, quartiles[2], min_duration)

        back = remove_last_inhales(front[0], front[1], candidate, quartiles[1], min_duration)

        output_segments.append(back)

    if len(output_segments) > 0 and output_segments[0] == [0, 0]:
        output_segments.pop(0)

    return output_segments


def remove_initial_inhales(start, end, amplitudes, threshold, min_duration):
    offset = next((index for index, val in enumerate(amplitudes) if val > threshold), 0)
    if min_duration < offset:
        offset = round(.75*offset)
        return [start + offset, end]
    return [start, end]


def remove_last_inhales(start, end, amplitudes, threshold, min_duration):
    offset = next((index for index, val in enumerate(reversed(amplitudes)) if val > threshold), 0)
    if min_duration < offset:
        offset = round(.85*offset)
        return [start, end - offset]
    return [start, end]


def audiosegment_to_librosawav(audiosegment):
    # samples = audiosegment.get_array_of_samples()
    # new_sound = audiosegment._spawn(samples)
    # arr = np.array(samples, dtype=np.float32)
    # y, index = librosa.effects.trim(arr)

    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr


def get_speaker_segments(args, audio_file, segments):
    from resemblyzer import preprocess_wav, VoiceEncoder
    encoder = VoiceEncoder()
    speaker_embeds = []

    speaker_segments = []
    for start, end in segments:
        clip = audio_file[start:end]
        segment_npy = audiosegment_to_librosawav(clip)
        segment_wav = preprocess_wav(segment_npy)
        current_embed = encoder.embed_utterance(segment_wav)
        is_any_similar = False

        min_similarity = 0.75
        name_id = len(speaker_embeds)
        for index, speaker_embed in enumerate(speaker_embeds):
            similarity = current_embed @ speaker_embed

            if similarity > min_similarity:
                min_similarity = similarity
                name_id = index
                is_any_similar = True

        if not is_any_similar:
            speaker_embeds.append(current_embed)
        speaker_segments.append((name_id, [start, end]))

    return speaker_segments


def segment_file(args, filepath, index):
    sys.setrecursionlimit(15000)

    if not os.path.exists('{0}/{1}/{2}/'.format(args.media, args.name, args.output)):
        os.makedirs('{0}/{1}/{2}/'.format(args.media, args.name, args.output))

    file_basename = os.path.basename(filepath)

    segment_path = '{0}/{1}/{2}/{3}.segment.json'.format(args.media, args.name, args.output, file_basename)

    print("{}: {}".format(index, filepath))

    audio_file = AudioSegment.from_wav(filepath)

    segments = detect_nonsilent(audio_file, args.short_silence, args.silence_thresh)

    amplitudes = [chunk.dBFS for chunk in make_chunks(audio_file, 1)]

    is_expermental = True

    quartiles = np.percentile(amplitudes, [5, 10, 25])
    if not is_expermental:
        modified_segments = segments
    else:
        modified_segments = remove_inhales(segments, amplitudes, quartiles[0], args.short_silence,
                                           args.long_silence)

    modified_segments = get_speaker_segments(args, audio_file, modified_segments)

    print("{}: clips: {} 5% db:{} 10% db:{} 25% db:{} ".format(index, len(segments), quartiles[0], quartiles[1], quartiles[2]))

    json_dict = dict()
    json_dict["schemaVersion"] = "1.0"
    json_dict["sound_ranges"] = [{
        "short_silence": args.short_silence,
        "long_silence": args.long_silence,
        "silence_thresh": args.silence_thresh,
        "sounds": modified_segments
    }]

    json_dict["monologues"] = [{
            "start": start / 1000,
            "end": stop / 1000,
            "speaker": {
                "id": str(speaker_id),
                "name": None
            }
        } for speaker_id, (start, stop) in modified_segments]

    with open(segment_path, 'w') as outfile:
        json.dump(json_dict, outfile, indent=2)


def segment_many(args, filepaths):
    print('{}: Starting segment_many'.format(time.strftime("%H:%M:%S")))
    for index, filepath in filepaths:
        segment_file(args, filepath, index)
        print('Transcribed file {} of {} from "{}"'.format(index + 1, len(filepaths), filepath))




def segment_many_parallel(args, filepaths):
    from concurrent import futures

    max_workers = args.max_workers if args.max_workers > 0 else None

    print('{}: Starting segment_many_parallel'.format(time.strftime("%H:%M:%S")))
    with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for index, filepath in filepaths:
            future_result = pool.submit(segment_file, args, filepath, index)
            print('{}: Transcribed file {} of {} from "{}"'.format(time.strftime("%H:%M:%S", time.localtime()), index + 1, len(filepaths), filepath))


def main():
    parser = argparse.ArgumentParser(description='Segment audio based on silence.')
    parser.add_argument('--output', default='transcripts', help='output directory')
    parser.add_argument('--media', default='../Workspace', help='input directory')
    parser.add_argument('--name', required=True, help='name of project')
    parser.add_argument('--long_silence', default=1000, help='input directory')
    parser.add_argument('--short_silence', default=50, help='input directory')
    parser.add_argument('--min', default=2, type=int, help='min clip duration in seconds')
    parser.add_argument('--max', default=12, type=int, help='max clip duration in seconds')
    parser.add_argument('--reuse', dest='reuse', action='store_true', help='reuse transcripts')
    parser.add_argument('--silence_thresh', default=-55, help='silence threshold for silences(db)')
    parser.add_argument('--max_workers', default=1, type=int, help='silence threshold for silences(db)')
    args = parser.parse_args()

    input_dir = '{0}/{1}/wavs/16000'.format(args.media, args.name)

    filepaths = list(enumerate(sorted(glob.glob("{0}/*.wav".format(input_dir)), key=os.path.basename)))

    if args.max_workers is not 1:
        print("segment_many_parallel")
        segment_many_parallel(args, filepaths)
    else:
        print("segment_many")
        segment_many(args, filepaths)


if __name__ == "__main__":
    main()
