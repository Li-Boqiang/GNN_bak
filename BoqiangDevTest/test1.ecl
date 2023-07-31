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
              loss=tf.keras.losses.CategoricalCrossentropy,
              metrics=['accuracy'])
              ''';

// compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(),
//             loss='binary_crossentropy',
//             metrics=['AUC'])
//               ''';

// compileDef := '''compile(optimizer='SGD',
//             loss='binary_crossentropy',
//             metrics=['AUC'])
//               ''';

// compileDef := '''compile(optimizer=tf.keras.optimizers.Adam(),
//             loss='binary_crossentropy',
//             metrics=['AUC'])
//               ''';

// compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
//                loss=tf.keras.losses.categorical_crossentropy,
//                metrics=['accuracy'])
//               ''';

OUTPUT(ldef, NAMED('ldef')); 
                                           
OUTPUT(compileDef, NAMED('compileDef'));          
s := GNNI.GetSession();
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


//Get weight tensors
// Project

