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

  def Np2Tens(a, wi=0, maxSliceOverride=0, isWeights = False): 
    maxSliceLen = 250000
    epsilon = .000000001
    origShape = list(a.shape)
    flatA = a.reshape(-1)
    flatSize = flatA.shape[0]
    currSlice = 1
    indx = 0
    datType = dTypeDictR[str(a.dtype)]
    elemSize = dTypeSizeDict[datType]
    if maxSliceOverride:
      maxSliceSize = maxSliceOverride
    else:
      maxSliceSize = divmod(maxSliceLen, elemSize)[0]
    if isWeights and nNodes > 1 and flatSize > nNodes:
      altSliceSize = math.ceil(flatSize / nNodes)
      maxSliceSize = min([maxSliceSize, altSliceSize])
    while indx < flatSize:
      remaining = flatSize - indx
      if remaining >= maxSliceSize:
        sliceSize = maxSliceSize
      else:
        sliceSize = remaining
      dat = list(flatA[indx:indx + sliceSize])
      dat = [float(d) for d in dat]
      elemCount = 0
      for i in range(len(dat)):
        if abs(dat[i]) > epsilon:
          elemCount += 1
      if elemCount > 0 or currSlice == 1:
        if elemCount * (elemSize + 4) < len(dat):
          # Sparse encoding
          sparse = []
          for i in range(len(dat)):
            if abs(dat[i]) > epsilon:
              sparse.append((i, dat[i]))
          yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, [], sparse)
        else:
          # Dense encoding
          yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, dat, [])
      currSlice += 1
      indx += sliceSize

  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  assert 1==0, x_train.shape
  origShape = list(a.shape)
  return Np2Tens(x_train)
ENDEMBED;

// res := test();

// OUTPUT(res, NAMED('res'));
