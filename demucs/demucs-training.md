# Demucs Training Details

I checked demucs' repository for tests but I didn't find any, so I decided to write my own.
To understand the API details for training, I explored demucs' docs and tried to run the `test_pretrained` script.

## Steps I followed

I simply ran the `test_pretrained` script, but I got some package errors.
I then looked up this [training doc](https://github.com/adefossez/demucs/blob/main/docs/training.md) in the official repository, and figured out needed steps from here.

### Step 1 - Recreating env

I recereated the environment using `environment-cpu.yml`, and then I got an error for dataset path.

### Step 2 - Download musdb

Demucs uses `musdbhq` as the default dataset, I downloaded it from [here](https://zenodo.org/records/3338373).
The file size was large so I used `axel` for quicker download, and placed decompressed files inside `musdbhq` subfolder in my global datasets directory.
Based on the docs, I also updated the path in following files

- The `dset.musdb` key inside `conf/config.yaml`.
- The variable `MUSDB_PATH` inside `tools/automix.py`.

### Step 3 - Resolve ffmpeg paths

Had another bug related to `ffprobe`, which was related to me linking an older version of `ffmpeg` in `brew` for essentia. I unlinked and relinked `ffmpeg` to the latest formulae from brew, and then the `test_pretrained` script ran as intended.

### Step 4 - Exploring Dora The Explorer

Demucs uses [dora](https://github.com/facebookresearch/dora), the library from Meta, for experiment management.

- An XP is a unique set of hyperparameters with a given signature.
- The signature is hash of those hyperparameters, eg - `9357e12e`.
- Hash is obtained as a delta between the base config and config with command line overrides.

`dora` can be used to

- get hyperparameters from signature - `dora info -f 81de367c`
- run distributed training with hyperparameters - `dora run -d -f 81de367c`
- fine tuning - `dora run -d -f 81de367c continue_from=81de367c dset=auto_mus variant=finetune`
