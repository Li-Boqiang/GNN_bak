person := RECORD
  STRING20 name;
  UNSIGNED1 age;
END;

people := DATASET
  (
    [ 
      {'John', 30 },
      {'Jane', 25 },
      {'Tom', 35 }
    ], 
    person
  );

adults := TABLE
  (
    people, 
    {
      STRING20 name := name;
      UNSIGNED1 age := age;
      BOOLEAN isAdult := age >= 18;
    }
  );

OUTPUT(adults);