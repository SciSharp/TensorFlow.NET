using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.Summaries
{
    public class EventFileWriter
    {
        string _logdir;
        Queue<int> _event_queue;

        public EventFileWriter(string logdir, int max_queue = 10, int flush_secs= 120,
               string filename_suffix = null)
        {
            _logdir = logdir;
            Directory.CreateDirectory(_logdir);
            _event_queue = new Queue<int>(max_queue);
        }
    }
}
