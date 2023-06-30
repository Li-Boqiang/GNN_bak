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

// Prepare training data
RAND_MAX := POWER(2,32) -1;


// Test parameters
trainCount := 1000;
testCount := 100;
featureCount := 5;
classCount := 3;
numEpochs := 5;
batchSize := 128;


ldef := ['''layers.Dense(16, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(16, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

compileDef := '''compile(optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=0.05),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
              ''';


OUTPUT(ldef, NAMED('ldef')); 
mdef1 := DATASET(COUNT(ldef), TRANSFORM(kString, SELF.typ := kStrType.layer,
                                        SELF.id  := COUNTER,
                                        SELF.text := ldef[COUNTER]));  
OUTPUT(mdef1, NAMED('mdef_test'));                                             
OUTPUT(compileDef, NAMED('compileDef'));   


s := GNNI.GetSession(1);
OUTPUT(s, NAMED('s'));

mod := GNNI.DefineModel(s, ldef, compileDef);
OUTPUT(mod, NAMED('mod'));
wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));

NewWeights := PROJECT(wts, TRANSFORM(RECORDOF(LEFT), SELF.denseData := IF(LEFT.wi = 1, 
                [.5, .5, .5] + LEFT.densedata[4..], LEFT.densedata), SELF := LEFT));

OUTPUT(NewWeights, NAMED('NewWeights'));
mod2 := GNNI.SetWeights(mod, NewWeights);
wts2 := GNNI.GetWeights(mod2);
OUTPUT(wts2, NAMED('SetWeights'));


trainRec := RECORD
  UNSIGNED8 id;
  SET OF REAL4 x;
  SET OF REAL4 y;
END;

SET OF REAL4 targetFunc(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt0 := TANH(.5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5);
  rslt := MAP(rslt0 > -.25 => [1,0,0], rslt0 < .25 => [0,1,0], [0,0,1]);
  RETURN rslt;
END;

// Build the training data
train0 := DATASET(trainCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1],
                      SELF.y := [])
                      );

// Be sure to compute Y in a second step.  Otherewise, the RANDOM() will be executed twice and the Y will be based
// on different values than those assigned to X.  This is an ECL quirk that is not easy to fix.
train := PROJECT(train0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(train, NAMED('trainData'));

// Build the test data.  Same process as the training data.
test0 := DATASET(testCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := [])
                      );

test := PROJECT(test0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.
// Format as NumericField data.
trainX := NORMALIZE(train, featureCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.x[COUNTER]));
trainY := NORMALIZE(train, classCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.y[COUNTER]));

OUTPUT(trainX, NAMED('X1'));
OUTPUT(trainY, NAMED('y1'));

testX := NORMALIZE(test, featureCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.x[COUNTER]));
testY := NORMALIZE(test, classCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.y[COUNTER]));

OUTPUT(testX, NAMED('testX'));
OUTPUT(testY, NAMED('testY'));


mod3 := GNNI.FitNF(mod, trainX, trainY, batchSize := batchSize, numEpochs := numEpochs);

OUTPUT(mod3, NAMED('mod3'));

losses := GNNI.GetLoss(mod3);
OUTPUT(losses, NAMED('losses'));

metrics := GNNI.EvaluateNF(mod3, testX, testY);

OUTPUT(metrics, NAMED('metrics'));

preds := GNNI.PredictNF(mod3, testX);


OUTPUT(testY, ALL, NAMED('testDat'));
OUTPUT(preds, NAMED('predictions'));
