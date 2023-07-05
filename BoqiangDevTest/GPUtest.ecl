IMPORT PYTHON3 AS PYTHON;

STRING GPUtest() := EMBED(Python)
  import tensorflow as tf
  import sys
  import os
  import subprocess
  # return sys.version
  # return tf.__version__
  return str(tf.config.list_physical_devices('GPU'))
  #env_vars = os.environ
  #res = ""
  #for key, value in env_vars.items():
  #  res = res + f"{key}: {value}"
  #  res = res + '\n'
  #return res
  #if tf.test.is_gpu_available():
  #  return 'available'
  #else:
  #  return 'unavailable'

  command = "env"  # 替换为您要执行的实际命令
  # command = "nvcc --version"  # 替换为您要执行的实际命令
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  exit_code = result.returncode
  output = result.stdout
  error = result.stderr
  return output
  #t = tf.config.list_physical_devices('GPU')
  #return t[0]
ENDEMBED;

res := GPUtest();
OUTPUT(res, NAMED('res'))