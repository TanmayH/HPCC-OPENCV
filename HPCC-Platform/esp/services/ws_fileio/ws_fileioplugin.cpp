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

#pragma warning (disable : 4786)

#ifndef WsFileIO_API
#define WsFileIO_API DECL_EXPORT
#endif //WsFileIO_API

#include "ws_fileio_esp.ipp"

//ESP Bindings
#include "http/platform/httpprot.hpp"

//ESP Service
#include "ws_fileioservice.hpp"

#include "espplugin.hpp"

extern "C"
{
//when we aren't loading dynamically
// Change the function names when we stick with dynamic loading.
ESP_FACTORY IEspService * esp_service_factory(const char *name, const char* type, IPropertyTree *cfg, const char *process)
{
    if (strcmp(type, "WsFileIO")==0)
    {
        CWsFileIOEx* service = new CWsFileIOEx;
        service->init(cfg, process, name);
        return service;
    }
    return NULL;
}

ESP_FACTORY IEspRpcBinding * esp_binding_factory(const char *name, const char* type, IPropertyTree *cfg, const char *process)
{
    if (strcmp(type, "WsFileIO")==0)
    {
#ifdef _DEBUG
        http_soap_log_level log_level_ = hsl_all;
#else
        http_soap_log_level log_level_ = hsl_none;
#endif
        return new CWsFileIOSoapBindingEx(cfg, name, process, log_level_);
    }

    return NULL;
}

ESP_FACTORY IEspProtocol * esp_protocol_factory(const char *name, const char* type, IPropertyTree *cfg, const char *process)
{
    return http_protocol_factory(name, type, cfg, process);
}

};
