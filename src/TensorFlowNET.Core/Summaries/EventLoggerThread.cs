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

using Google.Protobuf;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Thread that logs events.
    /// </summary>
    public class EventLoggerThread
    {
        Queue<Event> _queue;
#pragma warning disable CS0414 // The field 'EventLoggerThread.daemon' is assigned but its value is never used
        bool daemon;
#pragma warning restore CS0414 // The field 'EventLoggerThread.daemon' is assigned but its value is never used
        EventsWriter _ev_writer;
        int _flush_secs;
        Event _sentinel_event;

        public EventLoggerThread(Queue<Event> queue, EventsWriter ev_writer, int flush_secs, Event sentinel_event)
        {
            daemon = true;
            _queue = queue;
            _ev_writer = ev_writer;
            _flush_secs = flush_secs;
            _sentinel_event = sentinel_event;
        }

        public void start() => run();

        public void run()
        {
            Task.Run(delegate
            {
                while (true)
                {
                    if (_queue.Count == 0)
                    {
                        Thread.Sleep(_flush_secs * 1000);
                        continue;
                    }

                    var @event = _queue.Dequeue();
                    _ev_writer._WriteSerializedEvent(@event.ToByteArray());
                    Thread.Sleep(1000);
                }
            });
        }
    }
}
