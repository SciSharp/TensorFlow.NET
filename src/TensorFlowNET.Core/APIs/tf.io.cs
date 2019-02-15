using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class tf
    {
        public static Tensor read_file(string filename, string name = "") => gen_io_ops.read_file(filename, name);

        public static gen_image_ops image => new gen_image_ops();
    }
}
