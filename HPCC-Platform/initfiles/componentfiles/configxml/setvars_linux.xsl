<?xml version="1.0" encoding="UTF-8"?>
<!--
################################################################################
#    HPCC SYSTEMS software Copyright (C) 2012 HPCC Systems®.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
################################################################################
-->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:fo="http://www.w3.org/1999/XSL/Format" xml:space="default">
<xsl:strip-space elements="*"/>
<xsl:output method="text" media-type="text/plain" encoding="US-ASCII"/>
<xsl:param name="process" select="'thor'"/>
<xsl:param name="clusterType"/>
<xsl:template match="text()"/>
<xsl:template match="/">
    <xsl:apply-templates select="Environment/Software/ThorCluster[@name=$process]"/>
</xsl:template>

<xsl:template match="ThorCluster">
    <xsl:variable name="computerName" select="@computer"/>
    <xsl:variable name="domainName"><xsl:call-template name="getDomain"><xsl:with-param name="computer" select="@computer"/></xsl:call-template></xsl:variable>
    <xsl:variable name="domainNode" select="/Environment/Hardware/Domain[@name=$domainName]"/>
    <xsl:variable name="thoruser"   select="$domainNode/@username"/>
    <xsl:variable name="thorpasswd" select="$domainNode/@password"/>
# setvars script generated by setvars_linux.xsl
#
# General settings
    <xsl:if test="@name">
export THORNAME=<xsl:value-of select="@name"/>
    </xsl:if>

    <xsl:if test="@nodeGroup">
export THORPRIMARY=<xsl:value-of select="@nodeGroup"/>
    </xsl:if>
    <xsl:if test="@valgrindOptions">
export valgrindOptions="<xsl:value-of select='@valgrindOptions'/>"
    </xsl:if>
export THORMASTER=<xsl:call-template name="getNetAddress">
                    <xsl:with-param name="computer" select="ThorMasterProcess/@computer"/>
                </xsl:call-template>
export THORMASTERPORT=<xsl:call-template name="setOrDefault">
                        <xsl:with-param name="attribute" select="@masterport"/>
                        <xsl:with-param name="default" select="'20000'"/>
                      </xsl:call-template>
export THORSLAVEPORT=<xsl:call-template name="setOrDefault">
                        <xsl:with-param name="attribute" select="@slaveport"/>
                        <xsl:with-param name="default" select="'20100'"/>
                     </xsl:call-template>
export localthorportinc=<xsl:call-template name="setOrDefault">
                            <xsl:with-param name="attribute" select="@localThorPortInc"/>
                            <xsl:with-param name="default" select="'20'"/>
                        </xsl:call-template>
export slavespernode=<xsl:call-template name="setOrDefault">
                        <xsl:with-param name="attribute" select="@slavesPerNode"/>
                        <xsl:with-param name="default" select="'1'"/>
                     </xsl:call-template>
export channelsperslave=<xsl:call-template name="setOrDefault">
                        <xsl:with-param name="attribute" select="@channelsPerSlave"/>
                        <xsl:with-param name="default" select="'1'"/>
                     </xsl:call-template>
export DALISERVER=<xsl:call-template name="getDaliServers">
                    <xsl:with-param name="daliServer" select="@daliServers"/>
                </xsl:call-template>
export localthor=<xsl:call-template name="setOrDefault">
                    <xsl:with-param name="attribute" select="@localThor"/>
                    <xsl:with-param name="default" select="'false'"/>
                </xsl:call-template>
export breakoutlimit=<xsl:call-template name="setOrDefault">
                        <xsl:with-param name="attribute" select="Storage/@breakoutLimit"/>
                        <xsl:with-param name="default" select="'3600'"/>
                     </xsl:call-template>
export refreshrate=<xsl:call-template name="setOrDefault">
                        <xsl:with-param name="attribute" select="Storage/@refreshRate"/>
                        <xsl:with-param name="default" select="'3'"/>
                     </xsl:call-template>
export autoSwapNode=<xsl:choose>
           <xsl:when test="SwapNode/@AutoSwapNode = 'true'">true</xsl:when>
           <xsl:otherwise>false</xsl:otherwise>
            </xsl:choose>
<!-- Following SSH elements are required, being empty is fine -->
export SSHidentityfile=<xsl:value-of select="SSH/@SSHidentityfile"/>
export SSHusername=<xsl:value-of select="SSH/@SSHusername"/>
export SSHpassword=<xsl:value-of select="SSH/@SSHpassword"/>
export SSHtimeout=<xsl:value-of select="SSH/@SSHtimeout"/>
export SSHretries=<xsl:value-of select="SSH/@SSHretries"/>
export SSHsudomount=<xsl:value-of select="SSH/@SSHsudomount"/>
</xsl:template><!--/Environment/Software/ThorCluster-->

<xsl:template name="getDaliServers">
    <xsl:param name="daliServer"/>
    <xsl:for-each select="/Environment/Software/DaliServerProcess[@name=$daliServer]/Instance">
        <xsl:call-template name="getNetAddress">
            <xsl:with-param name="computer" select="@computer"/>
        </xsl:call-template>
        <xsl:if test="string(@port) != ''">:<xsl:value-of select="@port"/></xsl:if>
        <xsl:if test="position() != last()">, </xsl:if>
    </xsl:for-each>
</xsl:template><!--getDaliServers-->

<xsl:template name="getNetAddress">
    <xsl:param name="computer"/>
    <xsl:value-of select="/Environment/Hardware/Computer[@name=$computer]/@netAddress"/>
</xsl:template><!--getNetAddress-->

<xsl:template name="getDomain">
    <xsl:param name="computer"/>
    <xsl:value-of select="/Environment/Hardware/Computer[@name=$computer]/@domain"/>
</xsl:template>

<xsl:template name="setOrDefault">
    <xsl:param name="attribute"/>
    <xsl:param name="default"/>
    <xsl:choose>
        <xsl:when test="string($attribute) != ''">
            <xsl:value-of select="string($attribute)"/>
        </xsl:when>
        <xsl:otherwise>
            <xsl:value-of select="$default"/>
        </xsl:otherwise>
    </xsl:choose>
</xsl:template>

</xsl:stylesheet>