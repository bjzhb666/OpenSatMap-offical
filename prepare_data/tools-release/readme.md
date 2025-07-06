# Data preparation for OpenSatMap Instance-level Line Detection Track
This code is used to prepare the data for our OpenSatMap Instance-level Line Detection Track.

Please download the data from the following link: https://huggingface.co/datasets/z-hb/OpenSatMap/tree/main

Then put the data in the following structure:
```shell
├── picuse20trainvaltest
└── tools-release
```
Then run the following command to prepare the data:
```shell
bash tools-release/run_all_release.sh
```

Note: You only need to change the path in the `run_all_release.sh` file to your own path.
