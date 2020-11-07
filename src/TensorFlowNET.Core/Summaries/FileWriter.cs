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

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Writes `Summary` protocol buffers to event files.
    /// </summary>
    public class FileWriter : SummaryToEventTransformer
    {
        EventFileWriter event_writer;

        public FileWriter(string logdir, Graph graph,
            int max_queue = 10, int flush_secs = 120, string filename_suffix = null,
            Session session = null)
        {
            if (session == null)
            {
                event_writer = new EventFileWriter(logdir, max_queue, flush_secs, filename_suffix);
            }
        }
    }
}
