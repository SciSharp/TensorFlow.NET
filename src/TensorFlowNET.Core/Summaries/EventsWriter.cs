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

using System.IO;

namespace Tensorflow.Summaries
{
    public class EventsWriter
    {
        string _file_prefix;

        public EventsWriter(string file_prefix)
        {
            _file_prefix = file_prefix;
        }

        public void _WriteSerializedEvent(byte[] event_str)
        {
            File.WriteAllBytes(_file_prefix, event_str);
        }
    }
}
