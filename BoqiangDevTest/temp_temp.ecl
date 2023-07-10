IMPORT ML_Core;

EXPORT RECORD TestResult := RECORD
  UNSIGNED2 data;
END;

EXPORT test() := FUNCTION
  IMPORT PYTHONMODULE ML_Core_Python;
  IMPORT PYTHONMODULE numpy;

  arr := numpy.array([[1, 2], [3, 4]]);
  result := ML_Core_Python.CreateDynamicRecord('TestResult');
  result.data := ML_Core_Python.ConvertArrayToDynamic(arr);

  RETURN result;
END;

res := test();

OUTPUT(res, NAMED('res'));
