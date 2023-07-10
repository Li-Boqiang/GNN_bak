IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT ML_Core AS mlc;

kString := iTypes.kString;
kStrType := iTypes.kStrType;
NumericField := mlc.Types.NumericField;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

SET get_train_X() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train[:10]
  return x_train.flatten().tolist()
ENDEMBED;

SET get_train_Y() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  y_train = y_train[:10]
  return y_train.flatten().tolist()
ENDEMBED;

train_X := get_train_X();
train_Y := get_train_Y();

OUTPUT(train_X, NAMED('train_X'));
OUTPUT(train_Y, NAMED('train_Y'));

t1Rec := RECORD
  REAL4 value;
END;

intpuRec := RECORD
  UNSIGNED8 id;
  REAL4 value;
END;

x1 := DATASET(train_X, t1Rec);
y1 := DATASET(train_Y, t1Rec);
x2 := PROJECT(x1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
y2 := PROJECT(y1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
OUTPUT(x2, NAMED('x2'));
OUTPUT(y2, NAMED('y2'));

x3 := PROJECT(x2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/784) + 1, TRUNCATE(LEFT.id%784/28) + 1, LEFT.id%28 + 1], SELF.value := LEFT.value));
y3 := PROJECT(y2, TRANSFORM(TensData, SELF.indexes := [LEFT.id + 1, 1], SELF.value := LEFT.value));

OUTPUT(x3, NAMED('x3'));
OUTPUT(y3, NAMED('y3'));

x := Tensor.R4.MakeTensor([0,28,28], x3);
y := Tensor.R4.MakeTensor([0, 1], y3);
OUTPUT(x, NAMED('x_tensor'));
OUTPUT(y, NAMED('y_tensor'));

