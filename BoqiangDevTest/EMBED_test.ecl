IMPORT Python3 AS Python;

INTEGER4 MySquare(INTEGER4 val) := EMBED(Python)
    return val * val
ENDEMBED;

OUTPUT(MySquare(5));    // 25