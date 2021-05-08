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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public StringsApi strings { get; } = new StringsApi();

        public class StringsApi
        {
            string_ops ops = new string_ops();

            /// <summary>
            /// Converts all uppercase characters into their respective lowercase replacements.
            /// </summary>
            /// <param name="input"></param>
            /// <param name="encoding"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor lower(Tensor input, string encoding = "", string name = null)
                => ops.lower(input: input, encoding: encoding, name: name);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="pattern"></param>
            /// <param name="rewrite"></param>
            /// <param name="replace_global"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor regex_replace(Tensor input, string pattern, string rewrite,
                bool replace_global = true, string name = null)
                => ops.regex_replace(input, pattern, rewrite,
                    replace_global: replace_global, name: name);

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

            /// <summary>
            /// String lengths of `input`.
            /// </summary>
            /// <param name="input"></param>
            /// <param name="name"></param>
            /// <param name="unit"></param>
            /// <returns></returns>
            public Tensor string_length(Tensor input, string name = null, string unit = "BYTE")
                => ops.string_length(input, name: name, unit: unit);

            public Tensor format(string template, Tensor[] inputs, string placeholder = "{}", int summarize = 3, string name = null)
                => ops.string_format(inputs, template: template, placeholder: placeholder, summarize: summarize, name: name);

            public RaggedTensor split(Tensor input, char sep = ' ', int maxsplit = -1, string name = null)
                => ops.string_split_v2(input, sep: sep.ToString(), maxsplit : maxsplit, name : name);

            public (RaggedTensor, RaggedTensor) unicode_decode_with_offsets(Tensor input, string input_encoding, 
                string errors = "replace", int replacement_char = 0xFFFD, 
                bool replace_control_characters = false, string name = null)
                => ops.unicode_decode_with_offsets(input, input_encoding, errors,
                    replacement_char: replacement_char,
                    replace_control_characters: replace_control_characters,
                    name: name);
        }
    }
}
