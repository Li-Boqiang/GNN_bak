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


STRING mod_name1 := 'Xception';
mdef1 := 'weights="imagenet"';
mod1 := GNNI.DefineKAModel(s, mod_name1, mdef1);
OUTPUT(GNNI.getSummary(mod1), NAMED('Xception'));

STRING mod_name2 := 'VGG16';
mdef2 := 'weights="imagenet"';
mod2 := GNNI.DefineKAModel(s, mod_name2, mdef2);
OUTPUT(GNNI.getSummary(mod2), NAMED('VGG16'));

STRING mod_name3 := 'VGG19';
mdef3 := 'weights="imagenet"';
mod3 := GNNI.DefineKAModel(s, mod_name3, mdef3);
OUTPUT(GNNI.getSummary(mod3), NAMED('VGG19'));

STRING mod_name4 := 'ResNet50';
mdef4 := 'weights="imagenet"';
mod4 := GNNI.DefineKAModel(s, mod_name4, mdef4);
OUTPUT(GNNI.getSummary(mod4), NAMED('ResNet50'));

STRING mod_name5 := 'ResNet50V2';
mdef5 := 'weights="imagenet"';
mod5 := GNNI.DefineKAModel(s, mod_name5, mdef5);
OUTPUT(GNNI.getSummary(mod5), NAMED('ResNet50V2'));



// OUTPUT(wts, , '~thor::weights.csv', OVERWRITE, CSV(HEADING));
// OUTPUT(wts[1], NAMED('InitWeights_1'));
// OUTPUT(wts[2], NAMED('InitWeights_2'));


// How to test predict

