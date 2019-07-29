/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2018 HPCC Systems(R).

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

package com.HPCCSystems;

import java.net.*;
import java.util.Hashtable;
import java.util.List;
import java.util.ArrayList;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.Throwable;

public class HpccClassLoader extends java.lang.ClassLoader
{
    private long bytecode;
    private int bytecodeLen;
    private native Class<?> defineClassForEmbed(int bytecodeLen, long bytecode, String name);
    private Hashtable<String, Class<?>> classes = new Hashtable<>();
    static private Hashtable<String, java.net.URLClassLoader> pathLoaders = new Hashtable<>();
    private java.net.URLClassLoader pathLoader;
    private HpccClassLoader(String classPath, ClassLoader parent, int _bytecodeLen, long _bytecode, String dllname)
    {
        super(parent);
        if (classPath != null && !classPath.isEmpty())
        {
            synchronized(pathLoaders)
            {
                pathLoader = pathLoaders.get(classPath);
                if (pathLoader == null)
                {
                    List<URL> urls = new ArrayList<>();
                    String[] paths = classPath.split(";");
                    for (String path : paths)
                    {
                        try
                        {
                            if (path.contains(":"))
                                urls.add(new URL(path));
                            else
                                urls.add(new URL("file:" + path));
                        }
                        catch (MalformedURLException E)
                        {
                            // Ignore any that we don't recognize
                            // System.out.print(E.toString());
                        }
                    }
                    pathLoader = new URLClassLoader(urls.toArray(new URL[urls.size()]));
                    pathLoaders.put(classPath, pathLoader);
                }
            }
        }
        System.load(dllname);
        bytecodeLen = _bytecodeLen;
        bytecode = _bytecode;
    }
    public synchronized Class<?> findClass(String className) throws ClassNotFoundException
    {
        Class<?> result = classes.get(className);
        if (result == null)
        {
            if (bytecodeLen != 0)
                result = defineClassForEmbed(bytecodeLen, bytecode, className.replace(".","/"));
            if ( result == null && pathLoader != null)
                result = pathLoader.loadClass(className);
            if (result == null)
                return super.findClass(className);
            classes.put(className, result);
        }
        return result; 
    }
    public static HpccClassLoader newInstance(String classPath, ClassLoader parent, int _bytecodeLen, long _bytecode, String dllname)
    {
        return new HpccClassLoader(classPath, parent, _bytecodeLen, _bytecode, dllname); 
    }
    
    public static String getSignature(Method m)
    {
        StringBuilder sb = new StringBuilder();
        if ((m.getModifiers() & Modifier.STATIC) == 0)
            sb.append('@');
        sb.append('(');
        for(Class<?> c : m.getParameterTypes())
        { 
            String sig=java.lang.reflect.Array.newInstance(c, 0).toString();
            sb.append(sig.substring(1, sig.indexOf('@')));
        }
        sb.append(')');
        if (m.getReturnType()==void.class)
            sb.append("V");
        else
        {
            String sig=java.lang.reflect.Array.newInstance(m.getReturnType(), 0).toString();
            sb.append(sig.substring(1, sig.indexOf('@')));
        }
        return sb.toString();
    }

    /* get signature for method with given name, so long as there is only one */
    public static String getSignature ( Class<?> clazz, String simpleName ) throws Exception
    {
        String ret = null; 
        Method[] methods = clazz.getMethods();
        for (Method m : methods)
        {
            if ((m.getModifiers() & Modifier.PUBLIC) != 0 && m.getName().equals(simpleName))
            {
                if (ret == null)
                    ret = getSignature(m);
                else
                    throw new Exception("Multiple signatures found");  // multiple matches
            }
        }
        return ret;
    }
}