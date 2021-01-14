from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import sys
import json
import time

from deepspeech import Model, version
from timeit import default_timer as timer

import glob
import os

from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent, detect_silence
from multiprocessing import Process


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["text"] = word
            each_word["type"] = "WORD"
            each_word["start_time_ms"] = round(word_start_time, 4) * 1000
            each_word["end_time_ms"] = round(word_start_time + word_duration, 4) * 1000

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def split_long_nonsilence(nonsilence, silence, min_duration, max_duration):
    output_ranges = nonsilence
    delta = 1
    while delta != 0:
        delta = 0
        nonsilent_ranges = []
        for start, stop in output_ranges:
            if stop - start > max_duration:
                matches = [x for x in silence if start + min_duration <= x[0] < x[1] <= stop - min_duration]
                matches.sort(key=lambda x: x[1] - x[0], reverse=True)
                if len(matches) > 0:
                    last_start = start
                    for match in matches:
                        # selected_terms = [x for x in terms if last_start <= x["start_time_ms"] <= match[0]]
                        nonsilent_ranges.append([last_start, match[0]])
                        last_start = match[1]
                        delta = delta + 1
                        break

                    nonsilent_ranges.append([last_start, stop])
                else:
                    nonsilent_ranges.append([start, stop])
            else:
                nonsilent_ranges.append([start, stop])
        output_ranges = nonsilent_ranges

    if output_ranges[0] == [0, 0]:
        output_ranges.pop(0)

    return output_ranges


def metadata_json(metadata):
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json_result


def get_segments(args, audio_file, segments_file):
    if os.path.exists(segments_file):
        with open(segments_file) as json_file:
            json_data = json.load(json_file)
            return json_data["sound_ranges"][0]["sounds"]
    else:
        long_nonsilence = detect_nonsilent(audio_file, min_silence_len=args.long_silence,
                                           silence_thresh=args.silence_thresh)

        silence = detect_silence(audio_file, min_silence_len=args.short_silence,
                                 silence_thresh=args.silence_thresh)

        gaps_silence = list(map(lambda x: [x[0] + args.short_silence / 2, x[1] - args.short_silence / 2],
                                detect_silence(audio_file, min_silence_len=2 * args.short_silence,
                                               silence_thresh=args.silence_thresh + 20)))

        nonsilence1 = split_long_nonsilence(long_nonsilence, silence, args.min * 1000, args.max * 1000)

        segments = split_long_nonsilence(nonsilence1, gaps_silence, args.min * 1000, args.max * 1000)
        return segments


def audiosegment_to_librosawav(audiosegment):
    samples = audiosegment.get_array_of_samples()
    arr = np.array(samples).astype(np.float32)
    return arr


def get_speaker_segments(args, audio_file, segments_file):
    if os.path.exists(segments_file):
        with open(segments_file) as json_file:
            json_data = json.load(json_file)
            return [(monologue["speaker"]["id"], monologue["start"]*1000, monologue["end"]*1000,) for monologue in json_data["monologues"]]
    else:
        from resemblyzer import preprocess_wav, VoiceEncoder
        encoder = VoiceEncoder()
        speaker_embeds = []

        segments = get_segments(args, audio_file, segments_file)

        speaker_segments = []
        for start, end in segments:
            clip = audio_file[start:end]
            segment_npy = audiosegment_to_librosawav(clip)
            segment_wav = preprocess_wav(segment_npy)
            current_embed = encoder.embed_utterance(segment_wav)
            is_any_similar = False

            min_similarity = 0.85
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


def transcribe_file(args, ds, filepath, index):
    sys.setrecursionlimit(15000)

    if not os.path.exists('{0}/{1}/{2}/'.format(args.media, args.name, args.output)):
        os.makedirs('{0}/{1}/{2}/'.format(args.media, args.name, args.output))

    file_basename = os.path.basename(filepath)

    transcript_path = '{0}/{1}/{2}/{3}.transcript.json'.format(args.media, args.name, args.output, file_basename)

    segment_path = '{0}/{1}/{2}/{3}.segment.json'.format(args.media, args.name, args.output, file_basename)

    print("{}: {}".format(index, filepath))

    audio_file = AudioSegment.from_wav(filepath)

    # segments = get_segments(args, audio_file, segment_path)
    sentences = []
    monologues = []

    json_data = dict()
    json_data["transcripts"] = [{
        "confidence": 0.0,
        "words": [],
    }]

    sec_silence = AudioSegment.silent(duration=1000)

    speaker_segments = get_speaker_segments(args, audio_file, segment_path)

    print("clips: {}".format(len(speaker_segments)))

    for speaker_id, start, stop in speaker_segments:
        c_start = max(start-100, start)
        c_stop = min(stop, stop+100)
        clip_audio_16000 = sec_silence + audio_file[c_start:c_stop] + sec_silence

        audio_npy = np.frombuffer(clip_audio_16000.raw_data, np.int16)

        transcription = metadata_json(ds.sttWithMetadata(audio_npy, 1))

        terms = transcription["transcripts"][0]["words"]

        sentence_list = [{
            "start_time_ms": c_start + max(0, term["start_time_ms"] - 1000),
            "end_time_ms": c_stop + max(0, term["end_time_ms"] - 1000),
            "text": term["text"],
            "type": "WORD",
        } for term in terms]

        ds_text = ' '.join(s["text"] for s in sentence_list)

        if ds_text == '':
            continue

        json_data["transcripts"][0]["words"].append(sentence_list)

        if len(sentence_list) > 0:
            monologue = dict()

            monologue_terms = []
            for word in sentence_list:
                term = dict()
                term["start"] = min(word["start_time_ms"], stop) / 1000
                term["end"] = min(word["end_time_ms"], stop) / 1000
                term["text"] = word["text"]
                term["type"] = word["type"]
                monologue_terms.append(term)

            monologue["speaker"] = dict()
            monologue["speaker"]["id"] = speaker_id
            monologue["terms"] = monologue_terms
            monologue["start"] = start / 1000
            monologue["end"] = stop / 1000
            text = ' '.join(s["text"] for s in monologue_terms)
            include = True

            if include:
                monologues.append(monologue)

                sentence = dict()
                sentence["start"] = start / 1000
                sentence["end"] = stop / 1000

                sentence["sentence"] = text
                sentences.append(sentence)

    json_data["transcripts"][0]["sentences"] = sentences
    json_data["transcripts"][0]["speaker_segments"] = speaker_segments

    json_dict = dict()
    json_dict["schemaVersion"] = "2.0"
    json_dict["transcripts"] = json_data["transcripts"]
    json_dict["monologues"] = monologues

    with open(transcript_path, 'w') as outfile:
        json.dump(json_dict, outfile, indent=2)


def transcribe_many(args, filepaths):
    ds = Model(args.model)

    if args.beam_width:
        ds.setBeamWidth(args.beam_width)

    if args.scorer:
        print('Loading scorer from files {}'.format(args.scorer), file=sys.stderr)
        scorer_load_start = timer()
        ds.enableExternalScorer(args.scorer)
        scorer_load_end = timer() - scorer_load_start
        print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

        if args.lm_alpha and args.lm_beta:
            ds.setScorerAlphaBeta(args.lm_alpha, args.lm_beta)

    if args.hot_words:
        print('Adding hot-words', file=sys.stderr)
        for word_boost in args.hot_words.split(','):
            word, boost = word_boost.split(':')
            ds.addHotWord(word, float(boost))

    for index, filepath in filepaths:
        # if index < 48:
        #     continue
        transcribe_file(args, ds, filepath, index)
        print('Transcribed file {} of {} from "{}"'.format(index + 1, len(filepaths), filepath))


# TODO: untested
def transcribe_many_parallel(args, filepaths):
    for index, filepath in filepaths:
        ds = Model(args.model)

        if args.beam_width:
            ds.setBeamWidth(args.beam_width)

        if args.scorer:
            print('Loading scorer from files {}'.format(args.scorer), file=sys.stderr)
            scorer_load_start = timer()
            ds.enableExternalScorer(args.scorer)
            scorer_load_end = timer() - scorer_load_start
            print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

            if args.lm_alpha and args.lm_beta:
                ds.setScorerAlphaBeta(args.lm_alpha, args.lm_beta)

        if args.hot_words:
            print('Adding hot-words', file=sys.stderr)
            for word_boost in args.hot_words.split(','):
                word, boost = word_boost.split(':')
                ds.addHotWord(word, float(boost))
        p = Process(target=transcribe_file, args=(args, ds, filepath, index))
        p.start()
        p.join()
        print('{}: Transcribed file {} of {} from "{}"'.format(time.strftime("%H:%M:%S", time.localtime()), index + 1, len(filepaths), filepath))


def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True, help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=False, help='Path to the external scorer file')
    parser.add_argument('--beam_width', type=int, help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float,
                        help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float,
                        help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--extended', required=False, action='store_true', help='Output string from extended metadata')
    parser.add_argument('--json', required=False, action='store_true',
                        help='Output json from metadata with timestamp of each word')
    parser.add_argument('--candidate_transcripts', type=int, default=3,
                        help='Number of candidate transcripts to include in JSON output')
    parser.add_argument('--hot_words', type=str, help='Hot-words and their boosts.')
    parser.add_argument('--output', default='transcripts', help='output directory')
    parser.add_argument('--media', default='../Workspace', help='input directory')
    parser.add_argument('--name', required=True, help='name of project')
    parser.add_argument('--long_silence', default=1000, help='input directory')
    parser.add_argument('--short_silence', default=250, help='input directory')
    parser.add_argument('--min', default=2, type=int, help='min clip duration in seconds')
    parser.add_argument('--max', default=5, type=int, help='max clip duration in seconds')
    parser.add_argument('--reuse', dest='reuse', action='store_true', help='reuse transcripts')
    # parser.add_argument('--punctuate', default=False, help='use puncuation')
    parser.add_argument('--silence_thresh', default=-65, help='silence threshold for silences(db)')
    parser.set_defaults(reuse=False)
    args = parser.parse_args()

    input_dir = '{0}/{1}/wavs/16000'.format(args.media, args.name)

    filepaths = list(enumerate(sorted(glob.glob("{0}/*.wav".format(input_dir)), key=os.path.basename)))

    transcribe_many(args, filepaths)


if __name__ == "__main__":
    main()
