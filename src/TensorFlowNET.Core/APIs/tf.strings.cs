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

namespace Tensorflow
{
    public partial class tensorflow
    {
        public StringsApi strings { get; } = new StringsApi();

        public class StringsApi
        {
            string_ops ops = new string_ops();

            /// <summary>
            /// Return substrings from `Tensor` of strings.
            /// </summary>
            /// <param name="input"></param>
            /// <param name="pos"></param>
            /// <param name="len"></param>
            /// <param name="name"></param>
            /// <param name="uint"></param>
            /// <returns></returns>
            public Tensor substr(Tensor input, int pos, int len,
                    string name = null, string @uint = "BYTE")
                => ops.substr(input, pos, len, @uint: @uint, name: name);

            public Tensor substr(string input, int pos, int len,
                    string name = null, string @uint = "BYTE")
                => ops.substr(input, pos, len, @uint: @uint, name: name);
        }
    }
}
