/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;

namespace Tensorflow
{
    public static class tf2
    {
        public static bool _force_enable;
        internal static int flag = 0;
        public static void enable()
        {
            _force_enable = true;
            flag = 1;
        }
        public static void disable()
        {
            _force_enable = false;
            flag = 1;
        }
        public static bool enabled()
        {
            if(flag == 0)
            {
                var data = "";
                ((System.Collections.Generic.Dictionary<string, string>)System.Environment.GetEnvironmentVariables()).TryGetValue("TF2_BEHAVIOR", out data);
                if(string.IsNullOrEmpty(data))
                    data = "0";
                return data != "0";
            }
            else
                return _force_enable;
        }
    }
}
