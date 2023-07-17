IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT ML_Core AS mlc;

s := GNNI.GetSession(0);
OUTPUT(s, NAMED('s'));
GPU := GNNI.isGPUAvailable();
OUTPUT(GPU, NAMED('GPU'))