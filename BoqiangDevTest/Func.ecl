EXPORT Func := MODULE
    EXPORT STRING func1(UNSIGNED4 num) := FUNCTION
        RETURN '111111';
    END;

    EXPORT STRING func2(UNSIGNED4 num) := FUNCTION
        t2 := func1(num);
        RETURN t2;
    END;

    EXPORT STRING toJson(UNSIGNED4 mod) := FUNCTION
        RETURN '111111';
    END;

    EXPORT STRING testFunction(UNSIGNED4 num) := FUNCTION
        t1 := toJson(num);
        RETURN t1;
    END;
END;

