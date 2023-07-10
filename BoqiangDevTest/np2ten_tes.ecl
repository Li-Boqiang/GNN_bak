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
OUTPUT(s, NAMED('s'));

DATASET(t_tensor) test() := EMBED(Python)
    import tensorflow as tf
    global nodeId, nNodes, maxSliceLen
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return Np2Tens(x_train)
ENDEMBED;

res := test();

OUTPUT(res, NAMED('res'));
