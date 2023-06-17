using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.CSharp;

namespace Tensorflow.CodeGen
{
    public class FunctionGenerator
    {
        public void AppendFunction(OpDef op, StringBuilder sb)
        {
            // TODO: add descriptions
            sb.Append("public static ");
            int outputArgsCount = op.OutputArg.Count;
            if (outputArgsCount == 0)
            {
                sb.Append("Operation ");
            }
            else if (outputArgsCount == 1 && string.IsNullOrEmpty(op.OutputArg[0].NumberAttr)
                && string.IsNullOrEmpty(op.OutputArg[0].TypeListAttr))
            {
                sb.Append("Tensor ");
            }
            else
            {
                sb.Append("Tensor[] ");
            }
            string funcName = Utils.ConvertToUnderscore(op.Name);
            var token = SyntaxFactory.ParseToken(funcName);
            if (token.IsKeyword())
            {
                funcName = $"_{funcName}";
            }
            sb.Append($" {funcName}(");

            // define args
            AppendArgs(op, sb);
            sb.Append(")\n{\n");

            // begin to write main body
            sb.AppendLine("var _ctx = tf.Context;");

            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var dynamicDefaultValues);
            // deal with dynamic default values.
            foreach(var (name, expr) in dynamicDefaultValues)
            {
                sb.AppendLine($"if({name} is null)");
                sb.AppendLine("{");
                sb.AppendLine($"{name} = {expr};");
                sb.AppendLine("}");
            }

            sb.AppendLine("if(_ctx.executing_eagerly()){");

            if(HasRefArgs(op))
            {
                var possibleRefArg = op.InputArg.FirstOrDefault(x => x.IsRef, null);
                sb.AppendLine($"throw new RuntimeError(\"{funcName} op does not support eager execution. Arg {possibleRefArg.Name} is a ref.\");");
            }
            else
            {
                sb.Append("try\n{\n");

                AppendFastPathExecute(op, sb);
                if (outputArgsCount == 0)
                {
                    sb.AppendLine("return null;");
                }
                else if (outputArgsCount == 1 && string.IsNullOrEmpty(op.OutputArg[0].NumberAttr)
                    && string.IsNullOrEmpty(op.OutputArg[0].TypeListAttr))
                {
                    sb.AppendLine("return _fast_path_result[0];");
                }
                else
                {
                    sb.AppendLine("return _fast_path_result;");
                }

                sb.AppendLine("}"); // try

                sb.Append("catch(NotOkStatusException ex1)\n{\n");
                sb.AppendLine("throw ex1;");
                sb.AppendLine("}"); // catch

                sb.Append("catch(InvalidArgumentError ex2)\n{\n");
                sb.AppendLine("throw ex2;");
                sb.AppendLine("}"); // catch

                sb.Append("catch(Exception)\n{\n");
                sb.AppendLine("}"); // catch

                sb.Append("try\n{\n");
                AppendEagerFallbackCall(op, sb);
                sb.AppendLine("}"); // try

                sb.Append("catch(Exception)\n{\n");
                sb.AppendLine("}"); // catch
            }

            sb.AppendLine("}"); // if

            foreach(var (name, type, value) in attrValueDic.Where(x => x.Item2 == "string"))
            {
                if(value != "NOVALUE")
                {
                    sb.AppendLine($"if({name} is null)");
                    sb.AppendLine("{");
                    sb.AppendLine($"{name} = {value};");
                    sb.AppendLine("}");
                }
            }

            // begin to use op helper.
            AppendOpHelperCall(op, sb);
            sb.AppendLine("var _result = _op.outputs;");

            // check if it needs to record gradient.
            sb.Append("if(_execute.must_record_gradient())\n{\n");
            sb.Append("object[] _attrs = new object[]{");
            foreach (var attr in op.Attr)
            {
                string attrRealName = attr.Name;
                if (SyntaxFactory.ParseToken(attrRealName).IsKeyword())
                {
                    attrRealName += "_";
                }
                if (attr.Type == "type")
                {
                    sb.Append($"\"{attr.Name}\", _op._get_attr_type(\"{attrRealName}\"), ");
                }
                else if (attr.Type == "int")
                {
                    sb.Append($"\"{attr.Name}\", _op._get_attr_int(\"{attrRealName}\"), ");
                }
                else if (attr.Type == "bool")
                {
                    sb.Append($"\"{attr.Name}\", _op._get_attr_bool(\"{attrRealName}\"), ");
                }
                else
                {
                    sb.Append($"\"{attr.Name}\", _op.get_attr(\"{attr.Name}\"), ");
                }
            }
            if (sb[sb.Length - 1] == ' ' && sb[sb.Length - 2] == ',')
            {
                sb.Remove(sb.Length - 2, 2);
            }
            sb.Append("};\n");
            sb.AppendLine($"_execute.record_gradient(\"{op.Name}\", _op.inputs, _attrs, _result);");

            sb.AppendLine("}"); // if

            if (outputArgsCount == 0)
            {
                sb.AppendLine("return _op;");
            }
            else if (outputArgsCount == 1 && string.IsNullOrEmpty(op.OutputArg[0].NumberAttr)
                && string.IsNullOrEmpty(op.OutputArg[0].TypeListAttr))
            {
                sb.AppendLine("return _result[0];");
            }
            else
            {
                sb.AppendLine("return _result;");
            }
            sb.AppendLine("}"); // body

            sb.AppendLine();

            AppendEagerFallbackDefinition(op, sb);
        }

        public void AppendArgs(OpDef op, StringBuilder sb)
        {
            foreach (var arg in op.InputArg)
            {
                string argName = arg.Name;
                var token = SyntaxFactory.ParseToken(argName);
                if (token.IsKeyword())
                {
                    argName = $"{argName}_";
                }
                if (!string.IsNullOrEmpty(arg.NumberAttr) || !string.IsNullOrEmpty(arg.TypeListAttr))
                {
                    sb.Append($"Tensors {argName}, ");
                }
                else
                {
                    sb.Append($"Tensor {argName}, ");
                }
            }
            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var dynamicDefaultValues);
            foreach (var (key, typeStr, value) in attrValueDic.Where(x => x.Item3 == "NOVALUE"))
            {
                var token = SyntaxFactory.ParseToken(key);
                string realKey = key;
                if (token.IsKeyword())
                {
                    realKey += "_";
                }
                sb.Append($"{typeStr} {realKey}, ");
            }
            foreach (var (key, typeStr, value) in attrValueDic.Where(x => x.Item3 != "NOVALUE"))
            {
                var token = SyntaxFactory.ParseToken(key);
                string realKey = key;
                if (token.IsKeyword())
                {
                    realKey += "_";
                }
                sb.Append($"{typeStr} {realKey} = {value}, ");
            }
            sb.Append($"string? name = null");
        }

        public void AppendFastPathExecute(OpDef op, StringBuilder sb)
        {
            sb.Append($"var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, \"{op.Name}\", name)");
            sb.Append("{ args = new object[]{ ");
            foreach (var arg in op.InputArg)
            {
                string attrArgName = arg.Name;
                if (SyntaxFactory.ParseToken(attrArgName).IsKeyword())
                {
                    attrArgName += "_";
                }
                sb.Append($"{attrArgName}, ");
            }
            if (sb[sb.Length - 1] == ' ' && sb[sb.Length - 2] == ',')
            {
                sb.Remove(sb.Length - 2, 2);
            }

            sb.Append("}, attrs = new Dictionary<string, object>(){ ");
            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var _);
            foreach (var (key, _, _) in attrValueDic)
            {
                sb.Append($"[\"{key}\"] = {key}, ");
            }

            if (sb[sb.Length - 1] == ' ' && sb[sb.Length - 2] == ',')
            {
                sb.Remove(sb.Length - 2, 2);
            }
            sb.Append("}});\n");
        }

        public void AppendEagerFallbackCall(OpDef op, StringBuilder sb)
        {
            string funcName = $"{Utils.ConvertToUnderscore(op.Name)}_eager_fallback";
            sb.Append($"return {funcName}(");
            foreach (var arg in op.InputArg)
            {
                string inputArgRealName = arg.Name;
                if (SyntaxFactory.ParseToken(inputArgRealName).IsKeyword())
                {
                    inputArgRealName += "_";
                }
                sb.Append($"{inputArgRealName}, ");
            }
            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var _);
            foreach (var (key, _, _) in attrValueDic)
            {
                string keyRealName = key;
                if (SyntaxFactory.ParseToken(keyRealName).IsKeyword())
                {
                    keyRealName += "_";
                }
                sb.Append($"{key}: {keyRealName}, ");
            }
            sb.Append("name: name, ctx: _ctx);\n");
        }

        public void AppendEagerFallbackDefinition(OpDef op, StringBuilder sb)
        {
            sb.Append("public static ");
            int outputArgsCount = op.OutputArg.Count;
            if (outputArgsCount == 0)
            {
                sb.Append("Operation ");
            }
            else if (outputArgsCount == 1 && string.IsNullOrEmpty(op.OutputArg[0].NumberAttr)
                && string.IsNullOrEmpty(op.OutputArg[0].TypeListAttr))
            {
                sb.Append("Tensor ");
            }
            else
            {
                sb.Append("Tensor[] ");
            }
            string opName = op.Name;
            string funcName = Utils.ConvertToUnderscore(op.Name);
            sb.Append($" {funcName}_eager_fallback(");
            AppendFallBackFunctionArgs(op, sb);
            sb.Append(")\n{\n");

            var possibleRefArg = op.InputArg.FirstOrDefault(x => x.IsRef, null);
            if (possibleRefArg is not null)
            {
                sb.AppendLine($"throw new RuntimeError($\"{funcName} op does not support eager execution." +
                    $" Arg '{possibleRefArg.Name}' is a ref.\");");
                sb.AppendLine("}"); // body
                return;
            }

            if(op.InputArg.Any(x => !string.IsNullOrEmpty(x.NumberAttr)))
            {
                sb.AppendLine("List<Tensor> _inputs_flat_list = new();");
                foreach (var arg in op.InputArg)
                {
                    string realArgName = arg.Name;
                    if (SyntaxFactory.ParseToken(realArgName).IsKeyword())
                    {
                        realArgName = $"{realArgName}_";
                    }
                    if (string.IsNullOrEmpty(arg.NumberAttr))
                    {
                        sb.AppendLine($"_inputs_flat_list.Add({realArgName});");
                    }
                    else
                    {
                        sb.AppendLine($"_inputs_flat_list.AddRange({realArgName});");
                    }
                }
                sb.AppendLine($"var _inputs_flat = _inputs_flat_list.ToArray();");
            }
            else
            {
                sb.Append("Tensor[] _inputs_flat = new Tensor[]{");
                foreach (var arg in op.InputArg)
                {
                    string realArgName = arg.Name;
                    if (SyntaxFactory.ParseToken(realArgName).IsKeyword())
                    {
                        realArgName = $"{realArgName}_";
                    }
                    sb.Append($"{realArgName}, ");
                }
                if (sb[sb.Length - 1] == ' ' && sb[sb.Length - 2] == ',')
                {
                    sb.Remove(sb.Length - 2, 2);
                }
                sb.Append("};\n");
            }

            sb.Append("object[] _attrs = new object[]{");
            foreach (var attr in op.Attr)
            {
                if (attr.Type == "type")
                {
                    bool found = false;
                    foreach (var arg in op.InputArg)
                    {
                        string realArgName = arg.Name;
                        if (SyntaxFactory.ParseToken(realArgName).IsKeyword())
                        {
                            realArgName = $"{realArgName}_";
                        }
                        if (arg.TypeAttr == attr.Name)
                        {
                            sb.Append($"\"{attr.Name}\", {realArgName}.dtype, ");
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        string attrRealName = attr.Name;
                        if (SyntaxFactory.ParseToken(attrRealName).IsKeyword())
                        {
                            attrRealName = $"{attrRealName}_";
                        }
                        sb.Append($"\"{attr.Name}\", {attrRealName}, ");
                    }
                }
                else if(attr.Type == "list(type)")
                {
                    if (op.InputArg.Any(x => x.TypeListAttr == attr.Name))
                    {
                        continue;
                    }
                }
                else if(attr.Type == "int" && op.InputArg.Any(x => x.NumberAttr == attr.Name))
                {
                    bool found = false;
                    foreach (var arg in op.InputArg)
                    {
                        string realArgName = arg.Name;
                        if (SyntaxFactory.ParseToken(realArgName).IsKeyword())
                        {
                            realArgName = $"{realArgName}_";
                        }
                        if (arg.NumberAttr == attr.Name)
                        {
                            sb.Append($"\"{attr.Name}\", {realArgName}.Length, ");
                            found = true;
                            break;
                        }
                    }
                }
                else
                {
                    sb.Append($"\"{attr.Name}\", {attr.Name}, ");
                }
            }
            if (sb[sb.Length - 1] == ' ' && sb[sb.Length - 2] == ',')
            {
                sb.Remove(sb.Length - 2, 2);
            }
            sb.Append("};\n");

            sb.AppendLine($"var _result = _execute.execute(\"{op.Name}\", {outputArgsCount}, inputs: _inputs_flat, " +
                $"attrs: _attrs, ctx: ctx, name: name);");

            sb.Append("if(_execute.must_record_gradient())\n{\n");

            sb.AppendLine($"_execute.record_gradient(\"{op.Name}\", _inputs_flat, _attrs, _result);");

            sb.AppendLine("}"); // if

            if (outputArgsCount == 0)
            {
                sb.AppendLine("return null;");
            }
            else if (outputArgsCount == 1 && string.IsNullOrEmpty(op.OutputArg[0].NumberAttr)
                && string.IsNullOrEmpty(op.OutputArg[0].TypeListAttr))
            {
                sb.AppendLine("return _result[0];");
            }
            else
            {
                sb.AppendLine("return _result;");
            }

            sb.AppendLine("}"); // body
        }

        public void AppendFallBackFunctionArgs(OpDef op, StringBuilder sb)
        {
            foreach (var arg in op.InputArg)
            {
                string argName = arg.Name;
                var token = SyntaxFactory.ParseToken(argName);
                if (token.IsKeyword())
                {
                    argName = $"{argName}_";
                }
                if (!string.IsNullOrEmpty(arg.NumberAttr))
                {
                    sb.Append($"Tensors {argName}, ");
                }
                else
                {
                    sb.Append($"Tensor {argName}, ");
                }
            }
            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var _);
            foreach (var (key, typeStr, _) in attrValueDic)
            {
                var token = SyntaxFactory.ParseToken(key);
                string realKey = key;
                if (token.IsKeyword())
                {
                    realKey += "_";
                }
                sb.Append($"{typeStr} {realKey}, ");
            }
            sb.Append($"string name, Context ctx");
        }

        public void AppendOpHelperCall(OpDef op, StringBuilder sb)
        {
            sb.AppendLine("Dictionary<string, object> keywords = new();");
            foreach (var arg in op.InputArg)
            {
                string realArgName = arg.Name;
                if (SyntaxFactory.ParseToken(realArgName).IsKeyword())
                {
                    realArgName += "_";
                }
                sb.AppendLine($"keywords[\"{arg.Name}\"] = {realArgName};");
            }
            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var _);
            foreach (var (key, _, _) in attrValueDic)
            {
                sb.AppendLine($"keywords[\"{key}\"] = {key};");
            }
            sb.AppendLine($"var _op = tf.OpDefLib._apply_op_helper(\"{op.Name}\", name, keywords);");
        }

        private static bool HasRefArgs(OpDef op)
        {
            return op.InputArg.Any(x => x.IsRef);
        }
    }
}
