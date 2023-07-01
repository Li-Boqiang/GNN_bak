IMPORT PYTHON3 AS PYTHON;

STRING GPUtest() := EMBED(Python)
  import tensorflow as tf
  #gpu_device_name = tf.test.gpu_device_name()
  #assert 1==0, len(gpu_device_name)
  # assert 1==0, tf.test.is_gpu_available()
  if tf.test.is_gpu_available():
    return 'available'
  else:
    return 'unavailable'
ENDEMBED;

res := GPUtest();
OUTPUT(res, NAMED('res'))