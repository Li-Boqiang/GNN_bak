IMPORT Python3 AS Python;


EXPORT np2ten() := EMBED(Python)
  import numpy as np
  arr = np.array([[1, 2], [3, 4]])
  return arr.tolist()
ENDEMBED;
res := np2ten();



// DATASET(t_tensor) test() := EMBED(Python)
//   import tensorflow as tf
//   import numpy as np
//   def Np2Tens(a, wi=0, maxSliceOverride=0, isWeights = False): 
//     maxSliceLen = 250000
//     epsilon = .000000001
//     origShape = list(a.shape)
//     flatA = a.reshape(-1)
//     flatSize = flatA.shape[0]
//     currSlice = 1
//     indx = 0
//     dTypeDictR = {'float32':1, 'float64':2, 'int32':3, 'int64':4}
//     dTypeSizeDict = {1:4, 2:8, 3:4, 4:8}
//     datType = dTypeDictR[str(a.dtype)]
//     elemSize = dTypeSizeDict[datType]
//     if maxSliceOverride:
//       maxSliceSize = maxSliceOverride
//     else:
//       maxSliceSize = divmod(maxSliceLen, elemSize)[0]
//     if isWeights and nNodes > 1 and flatSize > nNodes:
//       altSliceSize = math.ceil(flatSize / nNodes)
//       maxSliceSize = min([maxSliceSize, altSliceSize])
//     while indx < flatSize:
//       remaining = flatSize - indx
//       if remaining >= maxSliceSize:
//         sliceSize = maxSliceSize
//       else:
//         sliceSize = remaining
//       dat = list(flatA[indx:indx + sliceSize])
//       dat = [float(d) for d in dat]
//       elemCount = 0
//       for i in range(len(dat)):
//         if abs(dat[i]) > epsilon:
//           elemCount += 1
//       if elemCount > 0 or currSlice == 1:
//         if elemCount * (elemSize + 4) < len(dat):
//           # Sparse encoding
//           sparse = []
//           for i in range(len(dat)):
//             if abs(dat[i]) > epsilon:
//               sparse.append((i, dat[i]))
//           yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, [], sparse)
//         else:
//           # Dense encoding
//           yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, dat, [])
//       currSlice += 1
//       indx += sliceSize

//   #mnist = tf.keras.datasets.mnist
//   #(x_train, y_train), (x_test, y_test) = mnist.load_data()
//   arr = np.array([[1, 2], [3, 4]])
  
//   return Np2Tens(arr)
// ENDEMBED;

