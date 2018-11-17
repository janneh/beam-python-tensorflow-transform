import argparse
import re
import logging
import time
import tempfile
import re as regex

from past.builtins import unicode

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

INPUT_FILE = 'datasets/kinglear.txt'
OUTPUT_FILE = 'outputs/kinglear_wordcount_{}'.format(time.time())

COUNTS_METADATA = dataset_metadata.DatasetMetadata(
  dataset_schema.from_feature_spec({
    'word': tf.FixedLenFeature([], tf.string),
    'count': tf.FixedLenFeature([], tf.int64)
  }))

class FindWords(beam.DoFn):
  def process(self, element):
    return regex.findall(r"[A-Za-z\']+", element)

class CountWordsTransform(beam.PTransform):
  def expand(self, p_collection):
    return (p_collection
      | "Find words" >> (beam.ParDo(FindWords()).with_input_types(unicode))
      | "Pair With One" >> beam.Map(lambda word: (word, 1))
      | "Group By" >> beam.GroupByKey()
      | "Aggregate Groups" >> beam.Map(lambda (word, ones): { 'word': word, 'count': sum(ones) }))

def run():
  pipeline_options = PipelineOptions(['--runner=DirectRunner'])
  pipeline_options.view_as(SetupOptions).save_main_session = True

  def preprocessing_fn(inputs):
    word = inputs['word']
    count = inputs['count']
    return {
      'word': word,
      'count': count,
      'count_normalized': tft.scale_to_0_1(count)
    }

  with beam.Pipeline(options=pipeline_options) as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      counts_data = (pipeline
      | "Load" >> ReadFromText(INPUT_FILE)
      | "Count Words" >> CountWordsTransform()
      )

      (transformed_data, transformed_metadata), transform_fn = (
      (counts_data, COUNTS_METADATA)
      | "AnalyzeAndTransform" >> tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

      column_names = ['word', 'count', 'count_normalized']
      transformed_data_coder = tft.coders.CsvCoder(column_names,transformed_metadata.schema)
      (transformed_data
      | "EncodeToCsv" >> beam.Map(transformed_data_coder.encode)
      | "Save" >> WriteToText(OUTPUT_FILE)
      )

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
