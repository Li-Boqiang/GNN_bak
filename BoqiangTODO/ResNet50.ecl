IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;

kString := iTypes.kString;
kStrType := iTypes.kStrType;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;


// load test data
imageRecord := RECORD
  STRING filename;
  DATA   image;   
       //first 4 bytes contain the length of the image data
  UNSIGNED8  RecPos{virtual(fileposition)};
END;

imageData := DATASET('~le::elephant',imageRecord,FLAT);
OUTPUT(imageData, NAMED('elephant'));

result := (STRING)(imageData[1].image);

SET OF REAL4 hexToNparry(DATA byte_array):= EMBED(Python)
  from PIL import Image
  import numpy as np
  import io
  try:
    import tensorflow as tf # V2.x
  except:
    assert 1 == 0, 'tensorflow not found'
  bytes_data = bytes(byte_array)
  image = Image.open(io.BytesIO(bytes_data))
  image = image.resize((224,224))
  I_array = np.array(image)
  I_array = tf.keras.applications.resnet50.preprocess_input(I_array)
  return I_array.flatten().tolist()
ENDEMBED;

t1Rec := RECORD
  REAL4 value;
END;

intpuRec := RECORD
  UNSIGNED8 id;
  REAL4 value;
END;

imageNpArray := hexToNparry(imageData[1].image);
OUTPUT(imageNpArray, NAMED('imageNpArray'));
x1 := DATASET(imageNpArray, t1Rec);
OUTPUT(x1, NAMED('x1'));
OUTPUT(COUNT(x1), NAMED('cnt_x1'));
x2 := PROJECT(x1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
OUTPUT(x2, NAMED('x2'));
OUTPUT(COUNT(x2), NAMED('cnt_x2'));

x3 := PROJECT(x2, TRANSFORM(TensData, SELF.indexes := [1, TRUNCATE(LEFT.id/(224*3)) + 1, TRUNCATE(LEFT.id/3)%224 + 1, LEFT.id%3 + 1], SELF.value := LEFT.value));

x := Tensor.R4.MakeTensor([0,224,224,3], x3);
OUTPUT(x, NAMED('x'));

// load model

s := GNNI.GetSession(1);
mdef := 'weights="imagenet"';
STRING modName := 'ResNet50';
mod := GNNI.DefineKAModel(s, modName, mdef);
OUTPUT(mod, NAMED('mod'));
summary := GNNI.getSummary(mod);
// OUTPUT(summary, NAMED('summary'));
res := GNNI.Predict(mod, x);

OUTPUT(res, NAMED('res'));



// OUTPUT(hexToNparry(imageData[1].image), NAMED('npShape'));

