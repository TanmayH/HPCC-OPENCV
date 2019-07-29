/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2012 HPCC Systems®.

    This program is free software: you can redistribute it and/or modify
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

x2 := '<Row><Pizza>Pepperoni</Pizza><Name><FName>Fred</FName><LName>Flintstone</LName></Name></Row>';
r := record
    string str;
end;

ds := dataset ([x2],r);

namesRec := RECORD
  STRING10  FName {xpath('FName')};
  STRING10  LName {xpath('LName')};
END;

namesRec2 := RECORD
    string Pizza {xpath('Pizza')};
    // DATASET(namesRec) Name {xpath('Name')};
  STRING10  FName {xpath('Name/FName')};
  STRING10  LName {xpath('Name/LName')};
END;

namesRec3 := RECORD
    namesRec2 fields;
END;

namesRec3    tTransform(ds L) := TRANSFORM
    namesRec2    lClaim    :=    fromxml(namesRec2, L.str, trim);
    self.fields                :=    lClaim;
    // self.lname                :=    lClaim.lname;
    // self.fname                :=    lClaim.fname;
    // self.pizza                 :=    lClaim.Pizza;
    // self := L;
END;

dsn := project (ds, tTransform(left));

output(dsn);
