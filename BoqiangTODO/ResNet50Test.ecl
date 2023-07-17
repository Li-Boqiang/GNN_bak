IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;

// hyperparam
batchSize := 128;
numEpochs := 5;
effNodes := 0;


imageRecord := RECORD
  STRING filename;
  DATA   image;   
       //first 4 bytes contain the length of the image data
  UNSIGNED8  RecPos{virtual(fileposition)};
END;

imageData := DATASET('~te::ele',imageRecord,FLAT);
OUTPUT(imageData, NAMED('elephant'));



// mnist_data_type := RECORD
// 	 INTEGER1 label;
// 	 DATA784 image;
// END;


// mnist_data_type_withid_x := RECORD
// 	 UNSIGNED id;   
// 	 SET Image;
// END;

// mnist_data_type_withid_y := RECORD
// 	 UNSIGNED id;     
//      INTEGER1 label;
// END;

// imageHight := 478;
// imageWidth := 379;
// imageChanel := 3;

// // Question: input a picture of elephant
// // how to convert hex to integer
// train0 := DATASET('~mnist::train', mnist_data_type, THOR);
// test0 := DATASET('~mnist::test', mnist_data_type, THOR);
// trainBig0 := DATASET('~mnist::big::train', mnist_data_type, THOR);
// testBig0 := DATASET('~mnist::big::test', mnist_data_type, THOR);


// SET byte2int(DATA784 image) := EMBED(Python)
//     import numpy as np
//     return np.asarray(image, dtype='B').astype('int').tolist()
// ENDEMBED;

// trainX1 := PROJECT(train0, TRANSFORM(mnist_data_type_withid_x, SELF.image:=byte2int(LEFT.image), SELF.id := COUNTER, SELF := LEFT));
// trainY1 := PROJECT(train0, TRANSFORM(mnist_data_type_withid_y, SELF.label:=LEFT.label, SELF.id := COUNTER, SELF:= LEFT));
// //output(trainX1);

// trainX2 := NORMALIZE(trainX1, imageHight*imageWidth*imageChanel, TRANSFORM(Tensor.R4.TensData,
//                             SELF.indexes := [LEFT.id, ((counter-1) div imageChanel) + 1, ((counter-1) % imageHight) +1],
//                             SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter])/127.5 -1));

// trainY2 := NORMALIZE(trainY1, 10, TRANSFORM(Tensor.R4.TensData,
//                             SELF.indexes := [LEFT.id, counter],
//                             SELF.value := IF(COUNTER = LEFT.label + 1,1,SKIP)));

// trainX3 := Tensor.R4.MakeTensor([0,28, 28], trainX2);
// trainY3 := Tensor.R4.MakeTensor([0, 10], trainY2);
// // output(trainX, named('trainX'));
// //output(trainY3, named('trainY3'));

// testX1 := PROJECT(test0, TRANSFORM(mnist_data_type_withid_x, SELF.image:=byte2int(LEFT.image), SELF.id := COUNTER, SELF := LEFT));
// testY1 := PROJECT(test0, TRANSFORM(mnist_data_type_withid_y, SELF.label:=LEFT.label, SELF.id := COUNTER, SELF:= LEFT));



// testX2 := NORMALIZE(testX1, 784, TRANSFORM(Tensor.R4.TensData,
//                             SELF.indexes := [LEFT.id, ((counter-1) div 28) + 1, ((counter-1) % 28) +1],
//                             SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter])/127.5 -1));

// testY2 := NORMALIZE(testY1, 10, TRANSFORM(Tensor.R4.TensData,
//                             SELF.indexes := [LEFT.id, counter],
//                             SELF.value := IF(COUNTER = LEFT.label + 1,1,SKIP)));
// testX3 := Tensor.R4.MakeTensor([0,28, 28], testX2);
// testY3 := Tensor.R4.MakeTensor([0, 10], testY2);

// s := GNNI.GetSession();

// STRING mod_name := 'ResNet50';
// mdef := 'weights="imagenet"';
// mod := GNNI.DefineKAModel(s, mod_name, mdef);
// wts := GNNI.GetWeights(mod);
// OUTPUT(GNNI.getSummary(mod), NAMED('ResNet50'));

// mod2 := GNNI.Fit(mod, trainX3, trainY3, batchSize := batchSize, numEpochs := numEpochs);
// OUTPUT(mod2, NAMED('mod2'));

// losses := GNNI.GetLoss(mod2);
// output(losses, NAMED('LOSSES'));
// metrics := GNNI.EvaluateMod(mod2, testX3, testY3);
// preds := GNNI.Predict(mod2, testX3);

// OUTPUT(testY3, ALL, NAMED('testDat'));
// OUTPUT(preds, NAMED('predictions'));