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
OUTPUT(ds, , '~thor::outdata.csv', OVERWRITE, CSV(HEADING(SINGLE)));

// OUTPUT(ds, , '~thor::outdata.csv', OVERWRITE, CSV(HEADING(1)));

// STD.File.Copy(OUTPUT_PATH, '~myuser/outputfile.csv', true);


myData := DATASET('~thor::outdata.csv', Layout, CSV);

// OUTPUT(myData, NAMED('myData'));