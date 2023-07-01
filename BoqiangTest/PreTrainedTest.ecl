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
STRING fname := 'ResNet50';


OUTPUT(fname, NAMED('fname'));
mdef := 'weights="imagenet"';




s := GNNI.GetSession(1);
mod := GNNI.DefineKAModel(s, fname, mdef);

OUTPUT(mod, NAMED('mod'));
wts := GNNI.GetWeights(mod);

OUTPUT(wts, , '~thor::weights.csv', OVERWRITE, CSV(HEADING));
OUTPUT(wts[1], NAMED('InitWeights_1'));
OUTPUT(wts[2], NAMED('InitWeights_2'));


// How to test predict

