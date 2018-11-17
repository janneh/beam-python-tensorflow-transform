### What it is

A beam pipeline that runs locally and counts the times a word appears in a file.

Output is a csv with columns: `word, count, count_normalized`

`count_normalized` is done by Tensorflow Transform, [`tft.scale_to_0_1`](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/scale_to_0_1)

### Install

```sh
pipenv install
```

### Run

```sh
pipenv run python pipeline.py
```
