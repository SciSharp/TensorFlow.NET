using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class FastPathOpExecInfo
    {
        public Context ctx { get; set; }
        public string device_name { get; set; }
        public string op_name { get; set; }
        public string name { get; set; }
        public object[] args { get; set; }
        public bool run_gradient_callback { get; set; }
        public bool run_post_exec_callbacks { get; set; }
        public bool run_callbacks { get; set; }
    }
}
