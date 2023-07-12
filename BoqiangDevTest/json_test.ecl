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

compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
              ''';


s := GNNI.GetSession(1);
OUTPUT(s, NAMED('s'));



mod := GNNI.DefineModel(s, ldef, compileDef);
OUTPUT(mod, NAMED('mod'));

json := GNNI.ToJSON(mod);

OUTPUT(json, NAMED('json'));

mod2 := GNNI.FromJSON(s, json);

OUTPUT(mod2, NAMED('mod2'));

