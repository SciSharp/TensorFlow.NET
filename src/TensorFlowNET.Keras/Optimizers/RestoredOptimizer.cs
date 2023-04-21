using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Saving;
using Tensorflow.Train;
using Tensorflow.Training;

namespace Tensorflow.Keras.Optimizers
{
    public class RestoredOptimizer: OptimizerV2, ITrackableWrapper, IKerasConfig
    {
        public String Identifier { get; } = "optimizer";
        public int Version { get; } = 2;
        public int MinConsumerVersion { get; } = 1;
        public int MinProducerVersion { get; } = 1;
        public RestoredOptimizer(): base(new ArgsDefinition.OptimizerV2Args() { Name = "RestoredOptimizer" })
        {
            _hypers_created = true;
        }

        public IKerasConfig get_config()
        {
            throw new NotImplementedException("Restoring functional Optimizers from SavedModels is not currently " +
                "supported. Please file a feature request if this limitation bothers you.");
        }

        public void SetValue(object name, object value)
        {
            if(name is not String str)
            {
                throw new TypeError($"The name of value to set must be string, but got {name.GetType()}");
            }
            if(value is Trackable trackable)
            {
                _track_trackable(trackable, str, overwrite: true);
            }
            if(value is IVariableV1 resource_variable)
            {
                if (!_hyper_variables.ContainsKey(str))
                {
                    _hyper_variables[str] = resource_variable;
                }
                else
                {
                    keras.backend.set_value(resource_variable, value);
                }
            }
            else if (value is float f)
            {
                _hyper[str] = f;
            }
            else
            {
                throw new NotImplementedException();
            }
        }
        
        public Trackable FromProto(SavedUserObject proto)
        {
            return new RestoredOptimizer();
        }
    }
}
