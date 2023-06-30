// 创建虚拟数据集
peopleRec := RECORD
  STRING5 name;
  UNSIGNED2 age;
  STRING10 job;
END;

people := DATASET([
  {'John', 25, 'Teacher'},
  {'Mary', 30, 'Engineer'},
  {'Bob', 35, 'Engineer'},
  {'Lisa', 40, 'Doctor'},
  {'Mike', 28, 'Teacher'},
  {'Anna', 32, 'Doctor'}
], peopleRec);

// 对数据集按job字段进行分组，并找出每组中的最大年龄
test := TABLE(people, {job, maxAge := MAX(GROUP, age)}, job);

// 不对数据集按job字段进行分组，找出所有人中的最大年龄
test2 := TABLE(people, {job, maxAge := MAX(GROUP, age)});

// 输出Age字段的最大值
test3 := MAX(people, age);
// 输出结果
OUTPUT(test, NAMED('MaxAgePerJob'));
OUTPUT(test2, NAMED('MaxAge'));
OUTPUT(test3, NAMED('Age'));