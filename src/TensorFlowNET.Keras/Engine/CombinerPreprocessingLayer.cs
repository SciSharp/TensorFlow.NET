using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Engine
{
    public class CombinerPreprocessingLayer : Layer
    {
        PreprocessingLayerArgs args;
        protected ICombiner combiner;
        protected bool _previously_updated;

        public CombinerPreprocessingLayer(PreprocessingLayerArgs args)
            : base(args)
        {
            _previously_updated = false;
        }

        public virtual void adapt(IDatasetV2 data, bool reset_state = true)
        {
            IAccumulator accumulator;
            if (!reset_state)
                accumulator = combiner.Restore();

            var next_data = data.make_one_shot_iterator();
            var data_element = next_data.next();
        }
    }
}
