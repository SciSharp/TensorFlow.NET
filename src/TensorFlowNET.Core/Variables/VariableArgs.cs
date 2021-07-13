using System;
using System.Collections.Generic;

namespace Tensorflow
{
    public class VariableArgs
    {
        public object InitialValue { get; set; }
        public Func<VariableArgs, IVariableV1> Getter { get; set; }
        public string Name { get; set; }
        public Shape Shape { get; set; }
        public TF_DataType DType { get; set; } = TF_DataType.DtInvalid;
        public IInitializer Initializer { get; set; }
        public bool Trainable { get; set; }
        public bool ValidateShape { get; set; } = true;
        public bool UseResource { get; set; } = true;
        public bool Overwrite { get; set; }
        public List<string> Collections { get; set; }
        public string CachingDevice { get; set; } = "";
        public VariableDef VariableDef { get; set; }
        public string ImportScope { get; set; } = "";
        public VariableSynchronization Synchronization { get; set; } = VariableSynchronization.Auto;
        public VariableAggregation Aggregation { get; set; } = VariableAggregation.None;
    }
}
