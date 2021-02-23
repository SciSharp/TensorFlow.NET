using System;
using System.Collections.Generic;
using Tensorflow.Contexts;

namespace Tensorflow
{
    public class FastPathOpExecInfo
    {
        public Context ctx { get; set; }
        public string device_name { get; set; }
        public string op_name { get; set; }
        public string name { get; set; }
        public object[] args { get; set; }
        public Dictionary<string, object> attrs { get; set; }
        public bool run_gradient_callback { get; set; }
        public bool run_post_exec_callbacks { get; set; }
        public bool run_callbacks { get; set; }
        public Action callbacks { get; set; }

        public FastPathOpExecInfo(string opName, string name, params object[] inputArgs)
        {
            this.op_name = opName;
            this.name = name;
            this.args = inputArgs;
        }
    }
}
