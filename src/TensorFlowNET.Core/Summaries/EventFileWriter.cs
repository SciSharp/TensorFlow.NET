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
using System.Collections.Generic;
using System.IO;

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Creates a `EventFileWriter` and an event file to write to.
    /// </summary>
    public class EventFileWriter
    {
        string _logdir;
        // Represents a first-in, first-out collection of objects.
        Queue<Event> _event_queue;
        EventsWriter _ev_writer;
        int _flush_secs;
        Event _sentinel_event;
#pragma warning disable CS0414 // The field 'EventFileWriter._closed' is assigned but its value is never used
        bool _closed;
#pragma warning restore CS0414 // The field 'EventFileWriter._closed' is assigned but its value is never used
        EventLoggerThread _worker;

        public EventFileWriter(string logdir, int max_queue = 10, int flush_secs = 120,
               string filename_suffix = null)
        {
            _logdir = logdir;
            Directory.CreateDirectory(_logdir);
            _event_queue = new Queue<Event>(max_queue);
            _ev_writer = new EventsWriter(Path.Combine(_logdir, "events"));
            _flush_secs = flush_secs;
            _sentinel_event = new Event();
            if (!string.IsNullOrEmpty(filename_suffix))
                // self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix)))
                throw new NotImplementedException("EventFileWriter filename_suffix is not null");
            _closed = false;
            _worker = new EventLoggerThread(_event_queue, _ev_writer, _flush_secs, _sentinel_event);
            _worker.start();
        }
    }
}
