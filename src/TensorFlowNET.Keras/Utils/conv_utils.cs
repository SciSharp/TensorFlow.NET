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

using System.Linq;

namespace Tensorflow.Keras.Utils
{
    public class conv_utils
    {
        public static string convert_data_format(string data_format, int ndim)
        {
            if (data_format == "channels_last")
                if (ndim == 3)
                    return "NWC";
                else if (ndim == 4)
                    return "NHWC";
                else if (ndim == 5)
                    return "NDHWC";
                else
                    throw new ValueError($"Input rank not supported: {ndim}");
            else if (data_format == "channels_first")
                if (ndim == 3)
                    return "NCW";
                else if (ndim == 4)
                    return "NCHW";
                else if (ndim == 5)
                    return "NCDHW";
                else
                    throw new ValueError($"Input rank not supported: {ndim}");
            else
                throw new ValueError($"Invalid data_format: {data_format}");
        }

        public static int[] normalize_tuple(int[] value, int n, string name)
        {
            if (value.Length == 1)
                return Enumerable.Range(0, n).Select(x => value[0]).ToArray();
            else
                return value;
        }

        public static string normalize_padding(string value)
        {
            return value.ToLower();
        }

        public static string normalize_data_format(string value)
        {
            if (string.IsNullOrEmpty(value))
                return ImageDataFormat.channels_last.ToString();
            return value.ToLower();
        }
    }
}
