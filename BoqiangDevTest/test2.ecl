Layout := RECORD
    STRING tableName;  
END;

TableNames := [
                        'tbl1',
                        'tbl2',
                        'tbl3',
                        'tbl4'
                     ];

ds_inlineLayout := DATASET(TableNames, {STRING tableName}); // Define the layout inline
ds_explicitLayout := DATASET(TableNames, Layout); // Use a an explicitly defined layout

OUTPUT(ds_inlineLayout);
OUTPUT(ds_explicitLayout);