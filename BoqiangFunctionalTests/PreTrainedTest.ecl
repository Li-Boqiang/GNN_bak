/*
About this test:
    Test the usability of Keras Applications
    reference: https://keras.io/api/applications/
*/

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


// STRING fname := 'MobileNetV2';       // passed
// STRING fname := 'ResNet50';          // passed
// STRING fname := 'VGG19';        // dataset too large to output to workunit, but output weights[1] works normally

s := GNNI.GetSession(1);
mdef := 'weights="imagenet"';

STRING mod_name1 := 'Xception';
mod1 := GNNI.DefineKAModel(s, mod_name1, mdef);
OUTPUT(GNNI.getSummary(mod1), NAMED('Xception'));

STRING mod_name2 := 'VGG16';
mod2 := GNNI.DefineKAModel(s, mod_name2, mdef);
OUTPUT(GNNI.getSummary(mod2), NAMED('VGG16'));

STRING mod_name3 := 'VGG19';
mod3 := GNNI.DefineKAModel(s, mod_name3, mdef);
OUTPUT(GNNI.getSummary(mod3), NAMED('VGG19'));

STRING mod_name4 := 'ResNet50';
mod4 := GNNI.DefineKAModel(s, mod_name4, mdef);
OUTPUT(GNNI.getSummary(mod4), NAMED('ResNet50'));

STRING mod_name5 := 'ResNet101';
mod5 := GNNI.DefineKAModel(s, mod_name5, mdef);
OUTPUT(GNNI.getSummary(mod5), NAMED('ResNet101'));

STRING mod_name6 := 'ResNet101V2';
mod6 := GNNI.DefineKAModel(s, mod_name6, mdef);
OUTPUT(GNNI.getSummary(mod6), NAMED('ResNet101V2'));

STRING mod_name7 := 'ResNet152';
mod7 := GNNI.DefineKAModel(s, mod_name7, mdef);
OUTPUT(GNNI.getSummary(mod7), NAMED('ResNet152'));

STRING mod_name8 := 'ResNet152V2';
mod8 := GNNI.DefineKAModel(s, mod_name8, mdef);
OUTPUT(GNNI.getSummary(mod8), NAMED('ResNet152V2'));

STRING mod_name9 := 'InceptionV3';
mod9 := GNNI.DefineKAModel(s, mod_name9, mdef);
OUTPUT(GNNI.getSummary(mod9), NAMED('InceptionV3'));

// STRING mod_name5 := 'InceptionResNetV2';
// mod5 := GNNI.DefineKAModel(s, mod_name5, mdef);
// OUTPUT(GNNI.getSummary(mod5), NAMED('ResNet50V2'));

// STRING mod_name5 := 'MobileNet';
// mod5 := GNNI.DefineKAModel(s, mod_name5, mdef);
// OUTPUT(GNNI.getSummary(mod5), NAMED('ResNet50V2'));

// STRING mod_name5 := 'MobileNetV2';
// mod5 := GNNI.DefineKAModel(s, mod_name5, mdef);
// OUTPUT(GNNI.getSummary(mod5), NAMED('ResNet50V2'));



// OUTPUT(wts, , '~thor::weights.csv', OVERWRITE, CSV(HEADING));
// OUTPUT(wts[1], NAMED('InitWeights_1'));
// OUTPUT(wts[2], NAMED('InitWeights_2'));


// How to test predict

