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

ds := dataset('ds', {integer m1; integer m2; }, THOR);

  r := record
    integer n;
  end;

  f(virtual dataset(r) d1, virtual dataset(r) d2) := count(d1(n=10))+count(d2(n=20));

result := f(ds{n:=m1}, ds{n:=m2});

result;
