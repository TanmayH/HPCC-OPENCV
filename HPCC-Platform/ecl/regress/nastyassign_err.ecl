/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2018 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

r1 := RECORD
    unsigned v;
    boolean n;
END;

r2 := RECORD
    r1 myVal;
    r1 myVal2;
    unsigned extra;
END;

r3 := RECORD
    unsigned myVal;
    unsigned extra;
END;


d3 := DATASET('d3', r3, thor);

r2 t(r3 l) := TRANSFORM
    SELF.myVal.v := 1;
//    SELF.myVal := ROW(transform(r1, SELF.v := 1; SELF.n := false));
    SELF := l;
    SELF.myVal2 := SELF.myVal;
END;

p := PROJECT(d3, t(LEFT));
output(p);
