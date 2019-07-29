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
#ifndef _HARDWARE_HPP_
#define _HARDWARE_HPP_

#include "EnvHelper.hpp"
#include "ComponentBase.hpp"

namespace ech
{

class Hardware : public ComponentBase
{
public:
   static const char* c_type;
   static const char* c_maker;
   static const char* c_speed;
   static const char* c_domain;
   static const char* c_os;

   Hardware(EnvHelper * envHelper);

   virtual void create(IPropertyTree *params);
   virtual unsigned add(IPropertyTree *params);
   virtual void modify(IPropertyTree *params);
   virtual void remove(IPropertyTree *params);

   const char* getComputerName(const char* netAddress);
   const char* getComputerNetAddress(const char* name);
   IPropertyTree* addComputer(IPropertyTree *params);
   IPropertyTree* addComputerType(IPropertyTree *params);
   IPropertyTree* addDomain(IPropertyTree *params);
   //IPropertyTree* addSwitch(IPropertyTree *params);

private:
   StringArray m_notifyUpdateList;
   StringArray m_notifyAddList;

};

}

#endif
