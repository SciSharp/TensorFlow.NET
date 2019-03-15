using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class control_flow_util
    {
        /// <summary>
        /// Return true if `op` is an Exit.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsLoopExit(Operation op)
        {
            return op.type == "Exit" || op.type == "RefExit";
        }

        /// <summary>
        /// Return true if `op` is a Switch.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsSwitch(Operation op)
        {
            return op.type == "Switch" || op.type == "RefSwitch";
        }
    }
}
