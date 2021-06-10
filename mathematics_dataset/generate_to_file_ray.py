from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate_settings
from mathematics_dataset import generate
import six
import ray
from six.moves import range

ray.init(object_store_memory=9000 * 1024 * 1024, num_cpus=4)

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write output text')
flags.DEFINE_boolean('train_split', False,
                     'Whether to split training data by difficulty')
flags.mark_flag_as_required('output_dir')


@ray.remote(memory=2500 * 1024 * 1024)
def write_qa(module_name, module, per_module):
  print('Begun processing module ', module_name)
  txt = ''
  for i in range(per_module):
    problem, _ = generate.sample_from_module(module)
    txt += (str(problem.question) + '\n')
    txt += (str(problem.answer) + '\n')
  print('Finished processing module ', module_name)
  return [module_name, txt]


def main(unused_argv):
  generate.init_modules(FLAGS.train_split)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if os.path.exists(output_dir):
    logging.fatal('output dir %s already exists', output_dir)
  os.makedirs(output_dir)
  for regime, flat_modules in six.iteritems(generate.filtered_modules):
    regime_dir = os.path.join(output_dir, regime)
    os.mkdir(regime_dir)
    per_module = generate.counts[regime]
    ids = [write_qa.remote(module_name, module, per_module) for module_name, module in six.iteritems(flat_modules)]
    outputs = ray.get(ids)
    for output in outputs:
      module_name = output[0]
      txt = output[1]
      path = os.path.join(regime_dir, module_name + '.txt')
      with open(path, 'w') as text_file:
        text_file.write(txt)

    #while len(ids):
    #  finished, rest = ray.wait(ids)
    #  result = finished[0]
    #  output = ray.get(result)
    #  module_name = output[0]
    #  txt = output[1]
    #  if module_name and txt:
    #    path = os.path.join(regime_dir, module_name + '.txt')
    #    with open(path, 'w') as text_file:
    #      text_file.write(txt)
    #    logging.info('Written %s', module_name)


if __name__ == '__main__':
  app.run(main)