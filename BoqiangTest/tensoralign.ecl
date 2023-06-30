IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;

t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

DATASET(t_Tensor) MakeTensor(UNSIGNED nRecs, UNSIGNED nRows, nCols, UNSIGNED wi) := FUNCTION
  DATASET(TensData) MakeData(UNSIGNED nRecs, UNSIGNED nRows, UNSIGNED nCols) := EMBED(Python)
    outrecs = []
    for i in range(nRecs):
        for row in range(nRows):
            for col in range(nCols):
                indx = [i+1,row+1, col+1]
                rec = (indx, 1.0)
                outrecs.append(rec)
    return outrecs
  ENDEMBED;
  tdat := MakeData(nRecs, nRows, nCols);
  tshape := [0, nRows, nCols]; // Zero first time implies record oriented tensor
  tens := Tensor.R4.MakeTensor(tshape, tdat, wi := wi);
  return tens;
END;

// Make 3 different sized tensors and then align them.
t1 := MakeTensor(10, 5, 5, 1);
t2 := MakeTensor(10, 15, 15, 2);
t3 := MakeTensor(10, 2, 3, 3);

t4 := t1 + t2 + t3;
OUTPUT(t1, NAMED('t1'));
OUTPUT(t2, NAMED('t2'));
OUTPUT(t3, NAMED('t3'));

OUTPUT(t4, NAMED('t4'));

combined := SORT(t1 + t2 + t3, wi, sliceId);


OUTPUT(combined, NAMED('Original'));

aligned := SORT(Tensor.R4.AlignTensors(combined), wi, sliceId);

OUTPUT(aligned, NAMED('Aligned'));
