using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.ArgsDefinition.Core;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Layers
{
    // A layer that uses `tf.einsum` as the backing computation.
    //   This layer can perform einsum calculations of arbitrary dimensionality.
    //   Args:
    //     equation: An equation describing the einsum to perform. This equation must
    //       be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
    //       `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
    //       expression sequence.
    //     output_shape: The expected shape of the output tensor (excluding the batch
    //       dimension and any dimensions represented by ellipses). You can specify
    //       None for any dimension that is unknown or can be inferred from the input
    //       shape.
    //     activation: Activation function to use. If you don't specify anything, no
    //       activation is applied (that is, a "linear" activation: `a(x) = x`).
    //     bias_axes: A string containing the output dimension(s) to apply a bias to.
    //       Each character in the `bias_axes` string should correspond to a character
    //       in the output portion of the `equation` string.
    //     kernel_initializer: Initializer for the `kernel` weights matrix.
    //     bias_initializer: Initializer for the bias vector.
    //     kernel_regularizer: Regularizer function applied to the `kernel` weights
    //       matrix.
    //     bias_regularizer: Regularizer function applied to the bias vector.
    //     activity_regularizer: Regularizer function applied to the output of the
    //       layer (its "activation").
    //     kernel_constraint: Constraint function applied to the `kernel` weights
    //       matrix.
    //     bias_constraint: Constraint function applied to the bias vector.
    //   Examples:
    //   **Biased dense layer with einsums**
    //   This example shows how to instantiate a standard Keras dense layer using
    //   einsum operations. This example is equivalent to
    //   `tf.keras.layers.Dense(64, use_bias=True)`.
    //   >>> layer = tf.keras.layers.EinsumDense("ab,bc->ac",
    //   ...                                     output_shape=64,
    //   ...                                     bias_axes="c")
    //   >>> input_tensor = tf.keras.Input(shape=[32])
    //   >>> output_tensor = layer(input_tensor)
    //   >>> output_tensor
    //   <... shape=(None, 64) dtype=...>
    //   **Applying a dense layer to a sequence**
    //   This example shows how to instantiate a layer that applies the same dense
    //   operation to every element in a sequence. Here, the `output_shape` has two
    //   values (since there are two non-batch dimensions in the output); the first
    //   dimension in the `output_shape` is `None`, because the sequence dimension `b`
    //   has an unknown shape.
    //   >>> layer = tf.keras.layers.EinsumDense("abc,cd->abd",
    //   ...                                     output_shape=(None, 64),
    //   ...                                     bias_axes="d")
    //   >>> input_tensor = tf.keras.Input(shape=[32, 128])
    //   >>> output_tensor = layer(input_tensor)
    //   >>> output_tensor
    //   <... shape=(None, 32, 64) dtype=...>
    //   **Applying a dense layer to a sequence using ellipses**
    //   This example shows how to instantiate a layer that applies the same dense
    //   operation to every element in a sequence, but uses the ellipsis notation
    //   instead of specifying the batch and sequence dimensions.
    //   Because we are using ellipsis notation and have specified only one axis, the
    //   `output_shape` arg is a single value. When instantiated in this way, the layer
    //   can handle any number of sequence dimensions - including the case where no
    //   sequence dimension exists.
    //   >>> layer = tf.keras.layers.EinsumDense("...x,xy->...y",
    //   ...                                     output_shape=64,
    //   ...                                     bias_axes="y")
    //   >>> input_tensor = tf.keras.Input(shape=[32, 128])
    //   >>> output_tensor = layer(input_tensor)
    //   >>> output_tensor
    //   <... shape=(None, 32, 64) dtype=...>
    //   
    public class EinsumDense : Layer
    {

        string equation;

        Activation activation;
        
        IVariableV1 bias;

        IVariableV1 kernel;

        string bias_axes;

        IInitializer kernel_initializer;

        IInitializer bias_initializer;

        System.Action kernel_constraint;

        System.Action bias_constraint;
        
        IRegularizer bias_regularizer;

        IRegularizer kernel_regularizer;

        Shape full_output_shape;
        
        Shape partial_output_shape;
        
        public EinsumDense(EinsumDenseArgs args) : base(args)
        {
            this.equation = args.Equation;
            this.partial_output_shape = args.OutputShape;
            this.bias_axes = args.BiasAxes;
            this.activation = args.Activation;
            this.kernel_initializer = args.KernelInitializer;
            this.bias_initializer = args.BiasInitializer;
            this.kernel_regularizer = args.KernelRegularizer;
            this.bias_regularizer = args.BiasRegularizer;
            this.kernel_constraint = args.KernelConstraint;
            this.bias_constraint = args.BiasConstraint;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            var shape_data = _analyze_einsum_string(this.equation, this.bias_axes, 
                input_shape.ToSingleShape(), this.partial_output_shape);
            var kernel_shape = shape_data.Item1;
            var bias_shape = shape_data.Item2;
            this.full_output_shape = shape_data.Item3;
            this.kernel = this.add_weight("kernel", shape: kernel_shape,
                                          initializer: this.kernel_initializer,
                                          regularizer: this.kernel_regularizer,
                                          //constraint: this.kernel_constraint,
                                          dtype: this.DType,
                                          trainable: true);
            if (bias_shape != null)
                this.bias = this.add_weight("bias", shape: bias_shape,
                                            initializer: this.bias_initializer,
                                            regularizer: this.bias_regularizer,
                                            //constraint: this.bias_constraint,
                                            dtype: this.DType,
                                            trainable: true);
            else
                this.bias = null;
            base.build(input_shape);
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return this.full_output_shape;
        }

        //public virtual object get_config() {
        //    var config = new Dictionary<object, object> {
        //        {
        //            "output_shape",
        //            this.partial_output_shape},
        //        {
        //            "equation",
        //            this.equation},
        //        {
        //            "activation",
        //            activations.serialize(this.activation)},
        //        {
        //            "bias_axes",
        //            this.bias_axes},
        //        {
        //            "kernel_initializer",
        //            initializers.serialize(this.kernel_initializer)},
        //        {
        //            "bias_initializer",
        //            initializers.serialize(this.bias_initializer)},
        //        {
        //            "kernel_regularizer",
        //            regularizers.serialize(this.kernel_regularizer)},
        //        {
        //            "bias_regularizer",
        //            regularizers.serialize(this.bias_regularizer)},
        //       {
        //            "activity_regularizer",
        //            regularizers.serialize(this.activity_regularizer)},
        //        {
        //            "kernel_constraint",
        //            constraints.serialize(this.kernel_constraint)},
        //        {
        //            "bias_constraint",
        //            constraints.serialize(this.bias_constraint)}};
        //    var base_config = base.get_config();
        //    return new dict(base_config.items().ToList() + config.items().ToList());
        //}

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var ret = tf.linalg.einsum(this.equation, (inputs, this.kernel.AsTensor()));
            if (this.bias != null)
                ret += this.bias.AsTensor();
            if (this.activation != null)
                ret = this.activation.Apply(ret);
            return ret;
        }
        /// <summary>
        /// Analyzes an einsum string to determine the required weight shape.
        /// </summary>
        public static (Shape, Shape, Shape) _analyze_einsum_string(string equation, string bias_axes, Shape input_shape, Shape output_shape)
        {
            var dot_replaced_string = Regex.Replace(equation, @"\.\.\.", "0");
            // This is the case where no ellipses are present in the string.
            var split_string = Regex.Match(dot_replaced_string, "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)");
            if (split_string.Success)
                return _analyze_split_string(split_string, bias_axes, input_shape, output_shape);
            // This is the case where ellipses are present on the left.
            split_string = Regex.Match(dot_replaced_string, "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)");
            if (split_string.Success)
                return _analyze_split_string(split_string, bias_axes, input_shape, output_shape, left_elided: true);
            // This is the case where ellipses are present on the right.
            split_string = Regex.Match(dot_replaced_string, "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0");
            if (split_string.Success)
                return _analyze_split_string(split_string, bias_axes, input_shape, output_shape);
            throw new ValueError($"Invalid einsum equation '{equation}'. " +
                $"Equations must be in the form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....");
        }

        /// <summary>
        /// Analyze an pre-split einsum string to find the weight shape.
        /// </summary>
        public static (Shape, Shape, Shape) _analyze_split_string(Match split_string,
                                                           string bias_axes,
                                                           Shape input_shape,
                                                           Shape output_shape,
                                                           bool left_elided = false)
        {
            List<int> bias_shape;
            Dictionary<char, int> output_dim_map;
            Dictionary<char, int> input_dim_map;

            var input_spec = split_string.Groups[1].Value;
            var weight_spec = split_string.Groups[2].Value;
            var output_spec = split_string.Groups[3].Value;
            var elided = input_shape.ndim - input_spec.Count();
            var _output_shape = new List<int>();
            _output_shape.Add((int)input_shape[0]);
            _output_shape.AddRange(output_shape.as_int_list());

            if (elided > 0 && left_elided)
                for (var i = 1; i < elided - 1; i++)
                    // We already inserted the 0th input dimension at dim 0, so we need to
                    // start at location 1 here.
                    _output_shape.Insert(1, (int)input_shape[i]);
            else if (elided > 0 && !left_elided)
                for (var i = input_shape.ndim - elided; i < input_shape.ndim - (input_shape.ndim - elided); i++)
                    _output_shape.Add((int)input_shape[i]);

            if (left_elided)
            {
                // If we have beginning dimensions elided, we need to use negative indexing
                // to determine where in the input dimension our values are.
                //input_dim_map = { dim: (i + elided) - len(input_shape) for i, dim in enumerate(input_spec) }
                input_dim_map = input_spec.Select((dim, i) => (i, dim))
                                          .ToDictionary(_ => _.dim, _ => _.i + elided - input_shape.ndim);
                // Because we've constructed the full output shape already, we don't need
                // to do negative indexing.
                //output_dim_map = { dim: (i + elided) for i, dim in enumerate(output_spec)}
                output_dim_map = output_spec.Select((dim, i) => (i, dim))
                                            .ToDictionary(_ => _.dim, _ => _.i + elided);
            }
            else
            {
                input_dim_map = input_spec.Select((dim, i) => (i, dim))
                                          .ToDictionary(_ => _.dim, _ => _.i);
                output_dim_map = output_spec.Select((dim, i) => (i, dim))
                                            .ToDictionary(_ => _.dim, _ => _.i);
            }

            foreach (var dim in input_spec)
            {
                var input_shape_at_dim = input_shape[input_dim_map[dim]];
                if (output_dim_map.TryGetValue(dim, out int index))
                {
                    var output_shape_at_dim = _output_shape[index];
                    if (output_shape_at_dim != -1 && output_shape_at_dim != input_shape_at_dim)
                        throw new ValueError($"Input shape and output shape do not match at shared dimension '{dim}'. " +
                                             $"Input shape is {input_shape_at_dim}, " +
                                             $"and output shape is {output_shape[output_dim_map[dim]]}.");
                }
            }

            foreach (var dim in output_spec)
            {
                if (!input_spec.Contains(dim) && !weight_spec.Contains(dim))
                {
                    throw new ValueError($"Dimension '{dim}' was specified in the output '{output_spec}' " +
                                         $"but has no corresponding dim in the input spec '{input_spec}' " +
                                         $"or weight spec '{output_spec}'");
                }
            }

            var weight_shape = new List<long>();
            foreach (var dim in weight_spec)
            {
                if (input_dim_map.ContainsKey(dim))
                    weight_shape.append(input_shape[input_dim_map[dim]]);
                else if (output_dim_map.ContainsKey(dim))
                    weight_shape.append(_output_shape[output_dim_map[dim]]);
                else throw new ValueError($"Weight dimension '{dim}' did not have a match in " +
                                          $"either the input spec '{input_spec}' " +
                                          $"or the output spec '{output_spec}'. " +
                                          $"For this layer, the weight must be fully specified.");
            }

            if (bias_axes != null)
            {
                var num_left_elided = left_elided ? elided : 0;
                var idx_map = output_spec.Select((_char, i) => (i, _char))
                                         .ToDictionary(_ => _._char, _ => _output_shape[_.i + num_left_elided]);
                foreach (var _char in bias_axes)
                    if (!output_spec.Contains(_char))
                        throw new ValueError($"Bias dimension '{_char}' was requested," +
                                             $" but is not part of the output spec '{output_spec}'");
                var first_bias_location = (from _char in bias_axes
                                           select output_spec.IndexOf(_char)).ToList().Min();
                var bias_output_spec = output_spec.Substring(first_bias_location);
                bias_shape = (from _char in bias_output_spec
                              select bias_axes.Contains(_char) ? idx_map[_char] : 1).ToList();
                if (!left_elided)
                    foreach (var _ in Enumerable.Range(0, elided))
                        bias_shape.append(1);
            }
            else bias_shape = null;

            return (weight_shape.ToArray(),
                   (bias_shape ?? new List<int>()).ToArray(),
                    _output_shape.ToArray());
        }
    }
}

    
