import functools
import multiprocessing
import os
import pathlib
import subprocess
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, Sequence, Tuple, List

import lmdb
import numpy as np
import torch
import yaml
import math
from absl import app, flags
from tqdm import tqdm
from udls.generated import AudioExample

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('input_path',
                          None,
                          help='Path to a directory containing audio files',
                          required=True)
flags.DEFINE_string('output_path',
                    None,
                    help='Output directory for the dataset',
                    required=True)
flags.DEFINE_integer('num_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('channels', 1, help="Number of audio channels")
flags.DEFINE_integer('sampling_rate',
                     44100,
                     help='Sampling rate to use during training')
flags.DEFINE_integer('max_db_size',
                     100,
                     help='Maximum size (in GB) of the dataset')
flags.DEFINE_multi_string(
    'ext',
    default=['aif', 'aiff', 'wav', 'opus', 'mp3', 'aac', 'flac', 'ogg'],
    help='Extension to search for in the input directory')
flags.DEFINE_bool('lazy',
                  default=False,
                  help='Decode and resample audio samples.')
flags.DEFINE_bool('dyndb',
                  default=True,
                  help="Allow the database to grow dynamically")
flags.DEFINE_bool('join_short_files',
                  default=False,
                  help="[vs fork] if files are smaller than num_signal and start+end with silence, use this to join them together before chunking")


def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()

class ConcatStream(object):
    def __init__(self, streams):
        self.streams = streams

    def read(self, n):
        hd, *tl = self.streams
        r = hd.read(n)
        # if chunk is incomplete
        if len(r) < n:
            # if there are more streams
            if len(tl):
                # continue the chunk from the next stream
                self.streams = tl
                r = b''.join((r, self.read(n-len(r))))
        # return the full size, or final, chunk
        return r

def load_audio_chunk(paths: List[str], n_signal: int,
                     sr: int, channels: int = 1,
                     ) -> Iterable[np.ndarray]:
    n_chan = None
    for path in paths:
        # print(path)
        _, input_channels = get_audio_channels(path)
        assert n_chan is None or n_chan==input_channels
        n_chan = input_channels

    channel_map = range(channels)
    if input_channels < channels:
        channel_map = (math.ceil(channels / input_channels) * list(range(input_channels)))[:channels]

    streams = []
    processes = []
    for i in range(channels): 
        channel_streams = []
        # loop over files and concatenate stdouts
        for path in paths:
            process = subprocess.Popen(
                [
                    'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path, 
                    '-ar', str(sr),
                    '-f', 's16le',
                    '-filter_complex', 'channelmap=%d-0'%channel_map[i],
                    '-'
                ],
                stdout=subprocess.PIPE,
            )
            channel_streams.append(process.stdout)
            processes.append(process)
        streams.append(ConcatStream(channel_streams))
    
    chunk = [s.read(n_signal * 4) for s in streams]
    while len(chunk[0]) == n_signal * 4:
        yield b''.join(chunk)
        chunk = [s.read(n_signal * 4) for s in streams]

    for process in processes:
        process.stdout.close()

def get_audio_length(path: str) -> float:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=duration'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        length = float(stdout)
        _, channels = get_audio_channels(path)
        return path, float(length), int(channels)
    except:
        return None
    
def get_audio_samples(path: str) -> float:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'stream=duration_ts'
            # 'stream'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    # print(stdout.decode())
    if process.returncode: return None
    try:
        # print(stdout.decode())
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        length = int(stdout)
        return path, length
    except:
        raise
        return None

def get_audio_channels(path: str) -> int:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'stream=channels'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        channels = int(stdout)
        return path, int(channels)
    except:
        return None 


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm

def get_metadata(audio_samples, channels: int = 1):
    audio = np.frombuffer(audio_samples, dtype=np.int16)
    audio = audio.astype(float) / (2**15 - 1)
    audio = audio.reshape(channels, -1)
    peak_amplitude = np.amax(np.abs(audio))
    rms_amplitude = np.sqrt(np.mean(audio**2))
    return {'peak': peak_amplitude, 'rms_amplitude': rms_amplitude}


def process_audio_array(audio: Tuple[int, bytes],
                        env: lmdb.Environment,
                        channels: int = 1) -> int:
    audio_id, audio_samples = audio
    buffers = {}
    buffers['waveform'] = AudioExample.AudioBuffer(
        shape=(channels, int(len(audio_samples) / channels)),
        sampling_rate=FLAGS.sampling_rate,
        data=audio_samples,
        precision=AudioExample.Precision.INT16,
    )

    ae = AudioExample(buffers=buffers)
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            ae.SerializeToString(),
        )
    return audio_id


def process_audio_file(audio: Tuple[int, Tuple[str, float]],
                       env: lmdb.Environment) -> int:
    audio_id, (path, length, channels) = audio
    ae = AudioExample(metadata={'path': path, 'length': str(length), 'channels': str(channels)})
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            ae.SerializeToString(),
        )
    return length


def flatmap(pool: multiprocessing.Pool,
            func: Callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


def search_for_audios(path_list: Sequence[str], extensions: Sequence[str]):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
            audios.append(p.rglob(f'*.{ext.upper()}'))
    audios = flatten(audios)
    return audios


def main(argv):
    if FLAGS.lazy and os.name in ["nt", "posix"]:
        while (answer := input(
                "Using lazy datasets on Windows/macOS might result in slow training. Continue ? (y/n) "
        ).lower()) not in ["y", "n"]:
            print("Answer 'y' or 'n'.")
        if answer == "n":
            print("Aborting...")
            exit()


    chunk_load = partial(load_audio_chunk,
                         n_signal=FLAGS.num_signal,
                         sr=FLAGS.sampling_rate,
                         channels=FLAGS.channels)

    output_dir = os.path.join(*os.path.split(FLAGS.output_path)[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # create database
    env = lmdb.open(
        FLAGS.output_path,
        map_size=FLAGS.max_db_size * 1024**3,
        map_async=not FLAGS.dyndb,
        writemap=not FLAGS.dyndb,
    )
    pool = multiprocessing.Pool()


    # search for audio files
    audios = search_for_audios(FLAGS.input_path, FLAGS.ext)
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    audios = [*audios]
    if len(audios) == 0:
        print("No valid file found in %s. Aborting"%FLAGS.input_path)


    assert not (FLAGS.join_short_files and FLAGS.lazy)
    if FLAGS.join_short_files:
        min_length = FLAGS.num_signal*2 # the miniumum chunk size
        # recover more data if the mininum file length is multiple chunks:
        joined_length = min_length * 32
        # check length of audio files and group them into lists
        audio_lengths = pool.imap_unordered(get_audio_samples, audios)
        cur_length, cur_paths = 0, []
        path_groups = []
        for path, length in audio_lengths:
            if cur_length < joined_length:
                cur_paths.append(path)
                cur_length += length
            else:
                path_groups.append(cur_paths)
                cur_paths = [path]
                cur_length = length
        if cur_length >= min_length:
            path_groups.append(cur_paths)
    else:
        path_groups = [[a] for a in audios]

    total_length = 0
    for _,length_s,_ in pool.imap_unordered(get_audio_length, audios):
        total_length += length_s
    print(f'total audio length: {int(total_length//60)}:{total_length%60}')

    if not FLAGS.lazy:
        # load chunks
        chunks = flatmap(pool, chunk_load, path_groups)
        chunks = enumerate(chunks)

        processed_samples = map(partial(process_audio_array, env=env, channels=FLAGS.channels), chunks)

        pbar = tqdm(processed_samples)
        n_seconds = 0
        for audio_id in pbar:
            n_seconds = (FLAGS.num_signal * 2) / FLAGS.sampling_rate * audio_id
            pbar.set_description(
                f'dataset length: {timedelta(seconds=n_seconds)}')
        pbar.close()
    else:
        audio_lengths = pool.imap_unordered(get_audio_length, audios)
        audio_lengths = filter(lambda x: x is not None, audio_lengths)
        audio_lengths = enumerate(audio_lengths)
        processed_samples = map(partial(process_audio_file, env=env),
                                audio_lengths)
        pbar = tqdm(processed_samples)
        n_seconds = 0
        for length in pbar:
            n_seconds += length
            pbar.set_description(
                f'dataset length: {timedelta(seconds=n_seconds)}')
        pbar.close()

    with open(os.path.join(
            FLAGS.output_path,
            'metadata.yaml',
    ), 'w') as metadata:
        yaml.safe_dump({'lazy': FLAGS.lazy, 'channels': FLAGS.channels, 'n_seconds': n_seconds, 'sr': FLAGS.sampling_rate}, metadata)
    pool.close()
    env.close()


if __name__ == '__main__':
    app.run(main)
