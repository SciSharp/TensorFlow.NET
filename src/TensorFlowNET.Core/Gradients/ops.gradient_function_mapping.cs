using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Gradients;

namespace Tensorflow
{
    public partial class ops
    {
        public static Func<Operation, Tensor[], Tensor[]> get_gradient_function(Operation op)
        {
            if (op.inputs == null) return null;

            // map tensorflow\python\ops\math_grad.py
            return (oper, out_grads) =>
            {
                // Console.WriteLine($"get_gradient_function: {oper.type} '{oper.name}'");

                switch (oper.type)
                {
                    case "Add":
                        return math_grad._AddGrad(oper, out_grads);
                    case "BiasAdd":
                        return nn_grad._BiasAddGrad(oper, out_grads);
                    case "Exp":
                        return math_grad._ExpGrad(oper, out_grads);
                    case "Identity":
                        return math_grad._IdGrad(oper, out_grads);
                    case "Log":
                        return math_grad._LogGrad(oper, out_grads);
                    case "MatMul":
                        return math_grad._MatMulGrad(oper, out_grads);
                    case "Merge":
                        return control_flow_grad._MergeGrad(oper, out_grads);
                    case "Mul":
                        return math_grad._MulGrad(oper, out_grads);
                    case "Mean":
                        return math_grad._MeanGrad(oper, out_grads);
                    case "Neg":
                        return math_grad._NegGrad(oper, out_grads);
                    case "Sum":
                        return math_grad._SumGrad(oper, out_grads);
                    case "Sub":
                        return math_grad._SubGrad(oper, out_grads);
                    case "Pow":
                        return math_grad._PowGrad(oper, out_grads);
                    case "RealDiv":
                        return math_grad._RealDivGrad(oper, out_grads);
                    case "Reshape":
                        return array_grad._ReshapeGrad(oper, out_grads);
                    case "Relu":
                        return nn_grad._ReluGrad(oper, out_grads);
                    case "Sigmoid":
                        return math_grad._SigmoidGrad(oper, out_grads);
                    case "Square":
                        return math_grad._SquareGrad(oper, out_grads);
                    case "Squeeze":
                        return array_grad._SqueezeGrad(oper, out_grads);
                    case "Softmax":
                        return nn_grad._SoftmaxGrad(oper, out_grads);
                    case "SoftmaxCrossEntropyWithLogits":
                        return nn_grad._SoftmaxCrossEntropyWithLogitsGrad(oper, out_grads);
                    case "Transpose":
                        return array_grad._TransposeGrad(oper, out_grads);
                    case "TopK":
                    case "TopKV2":
                        return nn_grad._TopKGrad(oper, out_grads);
                    default:
                        throw new NotImplementedException($"get_gradient_function {oper.type}");
                }
            };
        }
    }
}
