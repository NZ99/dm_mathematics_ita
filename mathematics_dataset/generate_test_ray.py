from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate_settings
from mathematics_dataset.modules import modules
import six
import ray
from six.moves import range

ray.init(num_cpus=4)

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write output text')
flags.DEFINE_boolean('train_split', False,
                     'Whether to split training data by difficulty')
flags.DEFINE_string('filter', '', 'restrict to matching module names')
flags.DEFINE_integer('per_train_module', 100, 'Num of examples per train module')
flags.DEFINE_integer('per_test_module', 10, 'Num of examples per test module')
flags.mark_flag_as_required('output_dir')



filtered_modules = collections.OrderedDict([])
counts = {}


def _make_entropy_fn(level, num_levels):
  """This returns a function that returns a subrange of entropy.

  E.g., if level=1 (medium) and num_levels=3, then the returned function will
  map the range [x, x + y] to [x + y/3, x + 2y/3].

  Args:
    level: Integer in range [0, num_levels - 1].
    num_levels: Number of difficulty levels.

  Returns:
    Function to restrict entropy range.
  """
  lower = level / num_levels
  upper = (level + 1) / num_levels
  def modify_entropy(range_):
    assert len(range_) == 2
    length = range_[1] - range_[0]
    return (range_[0] + lower * length, range_[0] + upper * length)
  return modify_entropy


def _filter_and_flatten(modules_):
  """Returns flattened dict, filtered according to FLAGS."""
  flat = collections.OrderedDict()

  def add(submodules, prefix=None):
    for key, module_or_function in six.iteritems(submodules):
      full_name = prefix + '__' + key if prefix is not None else key
      if isinstance(module_or_function, dict):
        add(module_or_function, full_name)
      else:
        if FLAGS.filter not in full_name:
          continue
        flat[full_name] = module_or_function

  add(modules_)

  # Make sure list of modules are in deterministic order. This is important when
  # generating across multiple machines.
  flat = collections.OrderedDict(
      [(key, flat[key]) for key in sorted(six.iterkeys(flat))])

  return flat


def init_modules(train_split=False):
  """Inits the dicts containing functions for generating modules."""
  if filtered_modules:
    return  # already initialized

  all_modules = collections.OrderedDict([])
  if train_split:
    all_modules['train-easy'] = modules.train(_make_entropy_fn(0, 3))
    all_modules['train-medium'] = modules.train(_make_entropy_fn(1, 3))
    all_modules['train-hard'] = modules.train(_make_entropy_fn(2, 3))
  else:
    all_modules['train'] = modules.train(_make_entropy_fn(0, 1))

  all_modules['interpolate'] = modules.test()
  all_modules['extrapolate'] = modules.test_extra()

  counts['train'] = FLAGS.per_train_module
  counts['train-easy'] = FLAGS.per_train_module // 3
  counts['train-medium'] = FLAGS.per_train_module // 3
  counts['train-hard'] = FLAGS.per_train_module // 3
  counts['interpolate'] = FLAGS.per_test_module
  counts['extrapolate'] = FLAGS.per_test_module

  for regime_, modules_ in six.iteritems(all_modules):
    filtered_modules[regime_] = _filter_and_flatten(modules_)


def sample_from_module(module):
  """Samples a problem, ignoring samples with overly long questions / answers.

  Args:
    module: Callable returning a `Problem`.

  Returns:
    Pair `(problem, num_dropped)`, where `problem` is an instance of `Problem`
    and `num_dropped` is an integer >= 0 indicating the number of samples that
    were dropped.
  """
  num_dropped = 0
  while True:
    problem = module()
    question = str(problem.question)
    if len(question) > generate_settings.MAX_QUESTION_LENGTH:
      num_dropped += 1
      continue
    answer = str(problem.answer)
    if len(answer) > generate_settings.MAX_ANSWER_LENGTH:
      num_dropped += 1
      continue
    return problem, num_dropped


@ray.remote
def write_qa(module_name, module, per_module):
  txt = ''
  for i in range(per_module):
    problem, _ = sample_from_module(module)
    txt += (str(problem.question) + '\n')
    txt += (str(problem.answer) + '\n')

  return [module_name, txt]


def main(unused_argv):
  init_modules(FLAGS.train_split)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if os.path.exists(output_dir):
    logging.fatal('output dir %s already exists', output_dir)
  os.makedirs(output_dir)
  for regime, flat_modules in six.iteritems(filtered_modules):
    regime_dir = os.path.join(output_dir, regime)
    os.mkdir(regime_dir)
    per_module = counts[regime]
    processes = [write_qa.remote(module_name, module, per_module) for module_name, module in six.iteritems(flat_modules)]
    outputs = ray.get(processes)
    for output in outputs:
      module_name = output[0]
      txt = output[1]
      path = os.path.join(regime_dir, module_name + '.txt')
      with open(path, 'w') as text_file:
        text_file.write(txt)
      logging.info('Written %s', module_name)


if __name__ == '__main__':
  app.run(main)