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
                Console.WriteLine($"get_gradient_function: {oper.type} '{oper.name}'");

                switch (oper.type)
                {
                    case "Add":
                        return math_grad._AddGrad(oper, out_grads);
                    case "Identity":
                        return math_grad._IdGrad(oper, out_grads);
                    case "Mul":
                        return math_grad._MulGrad(oper, out_grads);
                    case "Mean":
                        return math_grad._MeanGrad(oper, out_grads);
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
                    case "SoftmaxCrossEntropyWithLogits":
                        return nn_grad._SoftmaxCrossEntropyWithLogitsGrad(oper, out_grads);
                    default:
                        throw new NotImplementedException($"get_gradient_function {oper.type}");
                }
            };
        }
    }
}
