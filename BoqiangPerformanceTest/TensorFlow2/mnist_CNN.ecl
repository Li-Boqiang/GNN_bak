IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT ML_Core AS mlc;
IMPORT STD;

kString := iTypes.kString;
kStrType := iTypes.kStrType;
NumericField := mlc.Types.NumericField;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

// Test parameters
trainCount := 10000;
testCount := 1000;
featureCount := 5;
batchSize := 1024;
numEpochs := 10;
trainToLoss := .0001;
bsr := .25; // BatchSizeReduction.  1 = no reduction.  .25 = reduction to 25% of original.
lrr := 1.0;  // Learning Rate Reduction.  1 = no reduction.  .1 = reduction to 10 percent of original.


// Get training data
SET OF REAL4 get_train_X() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # x_train = x_train[:2]
  x_train = x_train*1.0/255
  return x_train.flatten().tolist()
ENDEMBED;

SET OF REAL4 get_train_Y() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # y_train = y_train[:2]
  y_one_hot = np.eye(10)[y_train]
  res = y_one_hot.flatten().tolist()
  return y_one_hot.flatten().tolist()
ENDEMBED;

train_X := get_train_X();
train_Y := get_train_Y();


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


x3 := PROJECT(x2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/784) + 1, TRUNCATE(LEFT.id%784/28) + 1, LEFT.id%28 + 1], SELF.value := LEFT.value));
y3 := PROJECT(y2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/10) + 1, LEFT.id%10 + 1], SELF.value := LEFT.value));


x := Tensor.R4.MakeTensor([0,28,28], x3);
y := Tensor.R4.MakeTensor([0, 10], y3);

// Define model
ldef := ['''layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28, 28, 1))''',
          '''layers.Conv2D(32, (5,5), padding='same', activation='relu')''',
          '''layers.MaxPool2D()''',
          '''layers.Dropout(0.25)''',
          '''layers.Conv2D(64, (3,3), padding='same', activation='relu')''',
          '''layers.Conv2D(64, (3,3), padding='same', activation='relu')''',
          '''layers.MaxPool2D(strides=(2,2))''',
          '''layers.Dropout(0.25)''',
          '''layers.Flatten()''',
          '''layers.Dense(128, activation='relu')''',
          '''layers.Dropout(0.5)''',
          '''layers.Dense(10, activation='softmax')'''];

compileDef := '''compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), 
                loss='categorical_crossentropy', metrics=['acc'])
              ''';

mdef1 := DATASET(COUNT(ldef), TRANSFORM(kString, SELF.typ := kStrType.layer,
                                        SELF.id  := COUNTER,
                                        SELF.text := ldef[COUNTER]));  
s := GNNI.GetSession(1);
mod := GNNI.DefineModel(s, ldef, compileDef);

// Train model
mod2 := GNNI.Fit(mod, x, y, batchSize := batchSize, numEpochs := numEpochs,
                      trainToLoss := trainToLoss, learningRateReduction := lrr,
                      batchSizeReduction := bsr);
losses := GNNI.GetLoss(mod2);

// Evaluate this model

// Get testing set
SET OF REAL4 get_test_X() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_test = x_test*1.0/255
  return x_test.flatten().tolist()
ENDEMBED;

SET OF REAL4 get_test_Y() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  y_one_hot = np.eye(10)[y_test]
  res = y_one_hot.flatten().tolist()
  return y_one_hot.flatten().tolist()
ENDEMBED;

x1_test := DATASET(train_X, t1Rec);
y1_test := DATASET(train_Y, t1Rec);
x2_test := PROJECT(x1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
y2_test := PROJECT(y1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));


x3_test := PROJECT(x2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/784) + 1, TRUNCATE(LEFT.id%784/28) + 1, LEFT.id%28 + 1], SELF.value := LEFT.value));
y3_test := PROJECT(y2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/10) + 1, LEFT.id%10 + 1], SELF.value := LEFT.value));


x_test := Tensor.R4.MakeTensor([0,28,28], x3);
y_test := Tensor.R4.MakeTensor([0, 10], y3);

metrics := GNNI.EvaluateMod(mod2, x_test, y_test);
preds := GNNI.Predict(mod2, x_test);

// OUTPUT results
ORDERED([OUTPUT(STD.Date.CurrentTime(TRUE), NAMED('startTime')), 
  OUTPUT(mod2, NAMED('mod2')),
  OUTPUT(STD.Date.CurrentTime(TRUE), NAMED('endTime')),
  OUTPUT(losses, NAMED('losses')),
  OUTPUT(metrics, NAMED('metrics')),
  OUTPUT(preds, NAMED('preds'))]);
