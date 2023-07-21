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

s := GNNI.GetSession(1);
mdef := 'weights="imagenet"';

STRING mod_name5 := 'ResNet101';
mod5 := GNNI.DefineKAModel(s, mod_name5, mdef);
OUTPUT(GNNI.getSummary(mod5), NAMED('ResNet101'));

