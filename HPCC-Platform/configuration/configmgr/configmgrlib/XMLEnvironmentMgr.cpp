/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.

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

#include "XMLEnvironmentMgr.hpp"
#include "XSDSchemaParser.hpp"
#include "XMLEnvironmentLoader.hpp"
#include "Exceptions.hpp"
#include "boost/version.hpp"


bool XMLEnvironmentMgr::createParser()
{
    m_pSchemaParser = std::make_shared<XSDSchemaParser>(m_pSchema);
    return true;
}


std::vector<std::shared_ptr<EnvironmentNode>> XMLEnvironmentMgr::doLoadEnvironment(std::istream &in, const std::shared_ptr<SchemaItem> &pSchemaItem, const std::string itemType)
{
    std::vector<std::shared_ptr<EnvironmentNode>> envNodes;
    try
    {
        XMLEnvironmentLoader envLoader;
        envNodes = envLoader.load(in, pSchemaItem, itemType);
    }
    catch (const std::exception &e)
    {
        std::string xmlError = e.what();
        std::string msg = "Unable to read/parse Environment file. Error = " + xmlError;
        throw (ParseException(msg));
    }
    return envNodes;
}


bool XMLEnvironmentMgr::serialize(std::ostream &out, const std::shared_ptr<EnvironmentNode> &pStartNode)
{
    pt::ptree envTree, topTree;
    serializeTree(envTree, pStartNode);
    topTree.add_child(pStartNode->getName(), envTree);

#if BOOST_VERSION >= 105700
    pt::write_xml(out, topTree, pt::xml_parser::xml_writer_make_settings<std::string>(' ', 4));
#else
    const char * myIndent = " ";
        pt::write_xml(out, topTree, pt::xml_parser::xml_writer_make_settings<char>(*myIndent , 4));
#endif

    return true;
}


bool XMLEnvironmentMgr::save(std::ostream &out)
{
    bool rc = true;
    try
    {
        pt::ptree envTree, topTree;
        serializeTree(envTree, m_pRootNode);
        topTree.add_child("Environment", envTree);
//        boost::property_tree::xml_writer_settings<std::string> settings;
        //pt::write_xml(out, topTree, pt::xml_parser::xml_writer_make_settings<std::string>(' ', 4));
#if BOOST_VERSION >= 105700
        pt::write_xml(out, topTree, pt::xml_parser::xml_writer_make_settings<std::string>(' ', 4));
#else
        const char * myIndent = " ";
        pt::write_xml(out, topTree, pt::xml_parser::xml_writer_make_settings<char>(*myIndent , 4));
#endif

    }
    catch (const std::exception &e)
    {
        std::string xmlError = e.what();
        m_message = "Unable to save Environment file. Error = " + xmlError;
        rc = false;
    }
    return rc;
}


void XMLEnvironmentMgr::serializeTree(pt::ptree &envTree, const std::shared_ptr<EnvironmentNode> &pEnvNode) const
{
    std::vector<std::shared_ptr<EnvironmentValue>> attributes;
    pEnvNode->getAttributes(attributes);
    for (auto attrIt = attributes.begin(); attrIt != attributes.end(); ++attrIt)
    {
        std::string attrValue;
        attrValue = (*attrIt)->getValue();

        if (!attrValue.empty() && !(*attrIt)->getSchemaValue()->isNoOutput())
        {
            envTree.put("<xmlattr>." + (*attrIt)->getName(), attrValue);
        }
    }
    std::shared_ptr<EnvironmentValue> pNodeValue = pEnvNode->getLocalEnvValue();
    if (pNodeValue)
    {
        envTree.put_value(pNodeValue->getValue());
    }
    std::vector<std::shared_ptr<EnvironmentNode>> children;
    pEnvNode->getChildren(children);
    for (auto childIt = children.begin(); childIt != children.end(); ++childIt)
    {
        pt::ptree nodeTree;
        serializeTree(nodeTree, *childIt);
        envTree.add_child((*childIt)->getName(), nodeTree);
    }
}
