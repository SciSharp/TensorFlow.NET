using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public record class LoadOptions
    {
        public bool allow_partial_checkpoint;
        public string experimental_io_device;
        public bool experimental_skip_checkpoint;
        public VariablePolicy experimental_variable_policy;

        public LoadOptions(bool allow_partial_checkpoint = false, string experimental_io_device = null,
            bool experimental_skip_checkpoint = false, string experimental_variable_policy = null)
        {
            this.allow_partial_checkpoint = allow_partial_checkpoint;
            this.experimental_io_device = experimental_io_device;
            this.experimental_skip_checkpoint = experimental_skip_checkpoint;
            this.experimental_variable_policy = VariablePolicy.from_obj(experimental_variable_policy);
        }
    }
}
