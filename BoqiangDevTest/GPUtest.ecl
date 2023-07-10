IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT ML_Core AS mlc;

STRING GPUtest() := EMBED(Python)
  import tensorflow as tf
  import os
  import subprocess
  # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
  return str(tf.config.list_physical_devices('GPU'))
ENDEMBED;

s := GNNI.GetSession(0);
OUTPUT(s, NAMED('s'));
res := GPUtest();
OUTPUT(res, NAMED('res'))