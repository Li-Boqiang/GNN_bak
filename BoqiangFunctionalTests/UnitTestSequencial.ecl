/*
About this test:
    Some basic unit tests for sequencial API.
*/

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

// Define layers
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

modSummary := GNNI.getSummary(mod);
OUTPUT(modSummary, NAMED('modSummary'));

wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));

NewWeights := PROJECT(wts, TRANSFORM(RECORDOF(LEFT), SELF.denseData := IF(LEFT.wi = 1, 
                [.5, .5, .5] + LEFT.densedata[4..], LEFT.densedata), SELF := LEFT));

OUTPUT(NewWeights, NAMED('NewWeights'));
mod2 := GNNI.SetWeights(mod, NewWeights);
wts2 := GNNI.GetWeights(mod2);
OUTPUT(wts2, NAMED('SetWeights'));

fullModel := GNNI.getModel(mod2);
OUTPUT(fullModel, NAMED('fullModel'));

mod3 := GNNI.setModel(s, fullModel);
OUTPUT(mod3, NAMED('mod3'));

mod3Summary := GNNI.getSummary(mod3);
OUTPUT(mod3Summary, NAMED('mod3Summary'));

