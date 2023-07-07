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

// // get data

// DATASET(NumericField) GetData() := EMBED(Python)
//   import tensorflow as tf

// ENDEMBED:

