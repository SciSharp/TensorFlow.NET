using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Gradients
{
    public class custom_gradient
    {
        public static string generate_name()
        {
            return $"CustomGradient-{ops.uid()}";
        }
    }
}
