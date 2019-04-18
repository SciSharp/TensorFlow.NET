using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Operations
{
    public class gen_nn_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor conv2d(object parameters)
        {
            var args = Python.ConvertToDict(parameters);

            var input = args["input"];
            var filter = args["filter"];
            var strides = args["strides"];
            var padding = args["padding"];
            var name = args["name"];
            var data_format = args.ContainsKey("data_format") ? args["data_format"] : "NHWC";
            var use_cudnn_on_gpu = args.ContainsKey("use_cudnn_on_gpu") ? args["use_cudnn_on_gpu"] : true;
            var dilations = args.ContainsKey("dilations") ? args["dilations"] : new int[] { 1, 1, 1, 1 };

            var _op = _op_def_lib._apply_op_helper("Conv2D", name: name?.ToString(), args: new
            {
                input,
                filter,
                strides,
                padding,
                use_cudnn_on_gpu,
                data_format,
                dilations
            });

            return _op.outputs[0];
        }

        public static Tensor bias_add(Tensor value,
            Tensor bias,
            string data_format = null,
            string name = null)
        {
            if (data_format == null)
                data_format = "NHWC";

            var _op = _op_def_lib._apply_op_helper("BiasAdd", name: name, args: new
            {
                value,
                bias,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor bias_add_grad(Tensor out_backprop,
            string data_format = "NHWC",
            string name = null)
        {
            if (data_format == null)
                data_format = "NHWC";

            var _op = _op_def_lib._apply_op_helper("BiasAddGrad", name: name, args: new
            {
                out_backprop,
                data_format
            });

            return _op.outputs[0];
        }

        public static Tensor[] _fused_batch_norm(Tensor x,
                Tensor scale,
                Tensor offset,
                Tensor mean,
                Tensor variance,
                float epsilon = 0.0001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FusedBatchNorm", name: name, args: new
            {
                x,
                scale,
                offset,
                mean,
                variance,
                epsilon,
                data_format,
                is_training
            });

            return _op.outputs;
        }

        public static Tensor log_softmax(Tensor logits, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("LogSoftmax", name: name, args: new
            {
                logits
            });

            return _op.outputs[0];
        }

        public static Tensor max_pool(Tensor input,
            int[] ksize,
            int[] strides,
            string padding,
            string data_format = "NHWC",
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("MaxPool", name: name, args: new
            {
                input,
                ksize,
                strides,
                padding,
                data_format,
            });

            return _op.outputs[0];
        }

        public static Tensor[] top_kv2(Tensor input, int k, bool sorted = true, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("TopKV2", name: name, args: new
            {
                input,
                k,
                sorted
            });

            return _op.outputs;
        }

        public static Tensor relu_grad(Tensor gradients, Tensor features, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ReluGrad", name: name, args: new
            {
                gradients,
                features
            });

            return _op.outputs[0];
        }

        public static Tensor softmax(Tensor logits, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Softmax", name: name, args: new
            {
                logits
            });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes softmax cross entropy cost and gradients to backpropagate.
        /// </summary>
        /// <param name="features"></param>
        /// <param name="labels"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("SoftmaxCrossEntropyWithLogits", name: name, args: new
            {
                features,
                labels
            });

            return (_op.outputs[0], _op.outputs[1]);
        }

        /// <summary>
        /// Computes rectified linear: `max(features, 0)`.
        /// </summary>
        /// <param name="features">A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `features`.</returns>
        public static Tensor relu(Tensor features, string name = null)
        {

            //_ctx = _context._context
            //if _ctx is not None and _ctx._eager_context.is_eager:
            //  try:
            //    _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
            //      _ctx._context_handle, _ctx._eager_context.device_name, "Relu", name,
            //      _ctx._post_execution_callbacks, features)
            //    return _result
            //  except _core._FallbackException:
            //    try:
            //      return relu_eager_fallback(
            //          features, name=name, ctx=_ctx)
            //    except _core._SymbolicException:
            //      pass  # Add nodes to the TensorFlow graph.
            //    except (TypeError, ValueError):
            //      result = _dispatch.dispatch(
            //            relu, features=features, name=name)
            //      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            //        return result
            //      raise
            //  except _core._NotOkStatusException as e:
            //    if name is not None:
            //      message = e.message + " name: " + name
            //    else:
            //      message = e.message
            //    _six.raise_from(_core._status_to_exception(e.code, message), None)
            //# Add nodes to the TensorFlow graph.
            //try:
            OpDefLibrary _op_def_lib = new OpDefLibrary();
            var _op = _op_def_lib._apply_op_helper("Relu", name: name, args: new { features });
            return _op.outputs[0];
            //except (TypeError, ValueError):
            //  result = _dispatch.dispatch(
            //        relu, features=features, name=name)
            //  if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            //    return result
            //  raise
            // var _result = _op.outputs.ToArray();
            //_inputs_flat = _op.inputs
            //_attrs = ("T", _op.get_attr("T"))
            //_execute.record_gradient(
            //    "Relu", _inputs_flat, _attrs, _result, name)
            //_result, = _result
            // return _result;
        }
    }
}
