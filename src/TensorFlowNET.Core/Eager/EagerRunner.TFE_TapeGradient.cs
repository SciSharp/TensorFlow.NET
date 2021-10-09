using System;
using Tensorflow.Gradients;
using Tensorflow.Util;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class EagerRunner
    {
        public Tensor[] TFE_TapeGradient(ITape tape,
            Tensor[] target,
            Tensor[] sources,
            Tensor[] output_gradients)
        {
            var target_vec = target;
            var sources_vec = sources;
            var sources_set = sources_vec;

            var seq_array = target;
            var source_tensors_that_are_targets = new UnorderedMap<Tensor, TapeTensor>();

            for (int i = 0; i < target.Length; ++i)
            {
                source_tensors_that_are_targets.Add(target_vec[i], new TapeTensor(seq_array[i]));
            }

            if (output_gradients != null)
            {
                throw new NotImplementedException("");
            }
            else
            {
                output_gradients = new Tensor[0];
            }

            var outgrad_vec = MakeTensorList(output_gradients);

            return tape.ComputeGradient(target_vec, sources_vec, source_tensors_that_are_targets, outgrad_vec);
        }

        Tensor[] MakeTensorList(Tensor[] tensors)
        {
            return tensors;
        }
    }
}
