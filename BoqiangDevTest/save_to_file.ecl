IMPORT STD;

Layout := RECORD
    STRING20 Name;
    UNSIGNED2 Age;
    STRING20 Occupation;
END;

ds := DATASET([{'John', 30, 'Doctor'},
               {'Alice', 25, 'Engineer'},
               {'Bob', 35, 'Teacher'}], Layout);

OUTPUT_PATH := 'outputfile.csv';

OUTPUT(ds, , '~thor::outdata.csv', OVERWRITE);

// STD.File.Copy(OUTPUT_PATH, '~myuser/outputfile.csv', true);


myData := DATASET('~thor::outdata.csv', Layout, CSV);
// myData := DATASET('~thor::outdata.csv', Layout, CSV(HEADING(1)));

OUTPUT(myData, NAMED('myData'));