/*****************************************************************************
   Copyright 2020 The TensorFlow.NET Authors. All Rights Reserved.

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

using Google.Protobuf;
using System.Text;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public CompatApi compat { get; } = new CompatApi();

        public class CompatApi
        {
            public CompatV1Api v1 { get; } = new CompatV1Api();

            internal string as_text(string bytes_or_text, Encoding? encoding = null)
            {
                if(encoding is null) encoding = Encoding.UTF8;
                return bytes_or_text;
            }
            internal string as_text(byte[] bytes_or_text, Encoding? encoding = null)
            {
                if(encoding is null) encoding = Encoding.UTF8;
                return encoding.GetString(bytes_or_text);
            }
            
            internal string as_str(string bytes_or_text, Encoding? encoding = null)
            {
                return as_text(bytes_or_text, encoding);
            }
            internal string as_str(byte[] bytes_or_text, Encoding? encoding = null)
            {
                return as_text(bytes_or_text, encoding);
            }

            public ByteString as_bytes(ByteString bytes, Encoding encoding = null)
            {
                return bytes;
            }
            public ByteString as_bytes(byte[] bytes, Encoding encoding = null)
            {
                return ByteString.CopyFrom(bytes);
            }
            public ByteString as_bytes(string text, Encoding encoding = null)
            {
                if(encoding is null)
                {
                    encoding = Encoding.UTF8;
                }
                return ByteString.CopyFrom(encoding.GetBytes(text));
            }
        }

        public bool executing_eagerly()
            => Context.executing_eagerly();
    }
}
