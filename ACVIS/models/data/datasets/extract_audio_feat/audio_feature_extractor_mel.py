"""Generate the per-frame VGGish mel arrays consumed by ACVIS."""

import argparse
import os

import numpy as np

import vggish_input
import vggish_params


def extract_subset(audio_root, subset, shard_index=0, num_shards=1):
    audio_dir = os.path.join(audio_root, subset, "WAVAudios")
    frame_dir = os.path.join(audio_root, subset, "JPEGImages")
    save_dir = os.path.join(audio_root, subset, "MELAudios")
    os.makedirs(save_dir, exist_ok=True)

    audio_files = sorted(
        name for name in os.listdir(audio_dir) if name.lower().endswith(".wav")
    )
    for index, filename in enumerate(audio_files, start=1):
        if (index - 1) % num_shards != shard_index:
            continue
        video_name = os.path.splitext(filename)[0]
        outfile = os.path.join(save_dir, video_name + ".npy")
        if os.path.exists(outfile):
            continue

        num_frames = len(os.listdir(os.path.join(frame_dir, video_name)))
        input_batch = vggish_input.wavfile_to_examples(
            os.path.join(audio_dir, filename), num_frames
        )
        np.testing.assert_equal(
            input_batch.shape,
            [num_frames, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS],
        )
        np.save(outfile, input_batch)
        print(
            "[{}/{}] {} -> {}".format(
                index, len(audio_files), filename, input_batch.shape
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", default="datasets")
    parser.add_argument(
        "--subsets", nargs="+", default=("train", "val", "test")
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    for subset in args.subsets:
        extract_subset(
            args.audio_root, subset, args.shard_index, args.num_shards
        )


if __name__ == "__main__":
    main()
