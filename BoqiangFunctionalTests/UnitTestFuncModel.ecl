/*
About this test:
    Some basic unit tests for Functional API.
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
FuncLayerDef := Types.FuncLayerDef;

// ldef1 is the sequential model definition for the Regression model (for reference)
ldef1 := ['''layers.Dense(256, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.Dense(1, activation=None)'''];
// ldef2 is the sequential model definition for the Classification model (for reference)
ldef2 := ['''layers.Dense(16, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(16, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

// fldef is the Functional model definition that combines the two above models
// The first field is the name of that layer.  The second is the definition of the layer.
// The third field is a list of predecessor layer names for this layer.  The
// Functional model defines a Directed Acyclic Graph (DAG), which is stitched together
// using the predecessor list.   If there were Concatenation layers, they would
// list multiple predecessors.  Note that Input layers have no predecessors.
fldef := DATASET([{'input1', '''layers.Input(shape=(5,))''', []},  // Regression Input
                {'d1', '''layers.Dense(256, activation='tanh')''', ['input1']}, // Regression Hidden 1
                {'d2', '''layers.Dense(256, activation='relu')''', ['d1']},   // Regression Hidden 2
                {'output1', '''layers.Dense(1, activation=None)''', ['d2']}, // Regression Output
                {'input2', '''layers.Input(shape=(5,))''', []}, // Classification Input
                {'d3', '''layers.Dense(16, activation='tanh', input_shape=(5,))''',['input2']}, // Classification Hidden 1
                {'d4', '''layers.Dense(16, activation='relu')''',['d3']}, // Classification Hidden 2
                {'output2', '''layers.Dense(3, activation='softmax')''', ['d4']}], // Classification Output
            FuncLayerDef);

compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.categorical_crossentropy],
              metrics=[])
              ''';

OUTPUT(fldef, NAMED('fldef')); 
OUTPUT(compileDef, NAMED('compileDef')); 

// Note that the order of the GNNI functions is maintained by passing tokens returned from
// one call into the next call that is dependent on it.
// For example, s is returned from GetSession().  It is used as the input to
// DefineModels(...) so
// that DefineModels() cannot execute until GetSession() has completed.
// Likewise, mod, the output from GetSession() is provided as input to Fit().  Fit in turn
// returns a token that is used by GetLoss(), EvaluateMod(), and Predict(),
// which are only dependent on Fit() having completed, and are not order
// dependent on one another.

// GetSession must be called before any other functions
s := GNNI.GetSession();
// DefineModel is dependent on the Session
//   fldef defines the functional model
//   inputs lists the input layer names
//   outputs lists the output layer names
//   compileDef contains the Keras compile statement.
mod := GNNI.DefineFuncModel(s, fldef, ['input1', 'input2'], ['output1', 'output2'], compileDef);

modSummary := GNNI.getSummary(mod);
OUTPUT(modSummary, NAMED('modSummary'));

// GetWeights returns the initialized weights that have been synchronized across all nodes.
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

// get mod3 's weights and compare with weights2
// get json and compare 

// compare the results