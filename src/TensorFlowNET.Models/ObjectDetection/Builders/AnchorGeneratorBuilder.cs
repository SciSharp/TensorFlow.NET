using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Models.ObjectDetection.Protos;
using static Tensorflow.Models.ObjectDetection.Protos.AnchorGenerator;

namespace Tensorflow.Models.ObjectDetection
{
    public class AnchorGeneratorBuilder
    {
        public AnchorGeneratorBuilder()
        {

        }

        public GridAnchorGenerator build(AnchorGenerator anchor_generator_config)
        {
            if(anchor_generator_config.AnchorGeneratorOneofCase == AnchorGeneratorOneofOneofCase.GridAnchorGenerator)
            {
                var grid_anchor_generator_config = anchor_generator_config.GridAnchorGenerator;
                return new GridAnchorGenerator(scales: grid_anchor_generator_config.Scales.Select(x => float.Parse(x.ToString())).ToArray());
            }
            throw new NotImplementedException("");
        }
    }
}
