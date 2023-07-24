STRING val1 := '1234';

SET OF STRING supportedModels := ['123', '124', '125'];


ASSERT(val1 IN supportedModels, 'Abc1');