IMPORT STD;
StartTime:= STD.Date.CurrentTime(TRUE); //Local Time

OUTPUT(StartTime, NAMED('StartTime'));

STD.System.Debug.Sleep(5000);
EndTime:= STD.Date.CurrentTime(TRUE); //Local Time

OUTPUT(EndTime, NAMED('EndTime'));
