IMPORT PYTHON3 AS PYTHON;

STRING GPUtest() := EMBED(Python)
  import tensorflow as tf
  import sys
  # return sys.version
  # return tf.__version__
  # return str(tf.config.list_physical_devices('GPU'))
  if tf.test.is_gpu_available():
    return 'available'
  else:
    return 'unavailable'
ENDEMBED;

res := GPUtest();
OUTPUT(res, NAMED('res'))