/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2012 HPCC Systems®.

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

//HOIST(dataset({unsigned i}) ds) := NOFOLD(SORT(NOFOLD(ds), i));
HOIST( ds) := MACRO
//NOFOLD(SORT(NOFOLD(ds), i))
//NOFOLD(ds)
ds
ENDMACRO;

mkRow(unsigned value) := TRANSFORM({ unsigned i }, SKIP(value = 1000000);   SELF.i := value);

dsOuter  := HOIST(DATASET([1,2,3], { unsigned i}, DISTRIBUTED));
dsInner1 := HOIST(DATASET([11,12,13], { unsigned i}, DISTRIBUTED));
dsInner2 := HOIST(DATASET([21,22,23], { unsigned i}, DISTRIBUTED));

innerSum1(unsigned x) := SUM(dsInner1(x != i), i);
outerSum1(unsigned x) := SUM(dsOuter(x != i), innerSum1(i));
innerSum2(unsigned x) := SUM(dsInner2(x != i), outerSum1(i));
outerSum2 := SUM(dsOuter, innerSum2(i));

sequential(
output(innerSum1(1));   // 36
output(outerSum1(1));   // 36 * 2 = 72
output(outerSum1(21));   // 36 * 3 = 108
output(innerSum2(1));   // 108 * 3 = 324
output(outerSum2);      // 324 * 3 = 972
);
