using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
            if (outputArgsCount > 1)
            {
                sb.Append("Tensor[] ");
            }
            else if (outputArgsCount == 1)
            {
                sb.Append("Tensor ");
            }
            else
            {
                sb.Append("Operation ");
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
                else if (outputArgsCount == 1)
                {
                    sb.AppendLine("return _fast_path_result[0];");
                }
                else
                {
                    sb.AppendLine("return _fast_path_result;");
                }

                sb.AppendLine("}"); // try

                sb.Append("catch(Exception)\n{\n");
                sb.AppendLine("}"); // catch

                sb.Append("try\n{\n");
                AppendEagerFallbackCall(op, sb);
                sb.AppendLine("}"); // try

                sb.Append("catch(Exception)\n{\n");
                sb.AppendLine("}"); // catch
            }

            sb.AppendLine("}"); // if

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
            else if (outputArgsCount == 1)
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
                if (!string.IsNullOrEmpty(arg.NumberAttr))
                {
                    sb.Append($"Tensors {argName}, ");
                }
                else
                {
                    sb.Append($"Tensor {argName}, ");
                }
            }
            var attrValueDic = GetAttrsDefaultValue(op);
            foreach (var (key, (typeStr, value)) in attrValueDic)
            {
                var token = SyntaxFactory.ParseToken(key);
                string realKey = key;
                if (token.IsKeyword())
                {
                    realKey += "_";
                }
                if (value != "NOVALUE")
                {
                    sb.Append($"{typeStr} {realKey} = {value}, ");
                }
                else
                {
                    sb.Append($"{typeStr} {realKey}, ");
                }
            }
            sb.Append($"string? name = null");
        }

        public void AppendFastPathExecute(OpDef op, StringBuilder sb)
        {
            sb.Append($"var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, \"{op.Name}\", name, ");
            foreach (var arg in op.InputArg)
            {
                string attrArgName = arg.Name;
                if (SyntaxFactory.ParseToken(attrArgName).IsKeyword())
                {
                    attrArgName += "_";
                }
                sb.Append($"{attrArgName}, ");
            }
            var attrValueDic = GetAttrsDefaultValue(op);
            foreach (var (key, _) in attrValueDic)
            {
                sb.Append($"\"{key}\", {key}, ");
            }
            if (sb[sb.Length - 1] == ' ' && sb[sb.Length - 2] == ',')
            {
                sb.Remove(sb.Length - 2, 2);
            }
            sb.Append("));\n");
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
            var attrValueDic = GetAttrsDefaultValue(op);
            foreach (var (key, _) in attrValueDic)
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
            sb.Append("public static Tensor");
            int outputArgsCount = op.OutputArg.Count;
            if (outputArgsCount > 1)
            {
                sb.Append("[]");
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

            sb.Append("object[] _attrs = new object[]{");
            var attrValueDic = GetAttrsDefaultValue(op);
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
                        if (attr.Name.StartsWith("T") && attr.Name.Length > 1)
                        {
                            string paramName = attr.Name.Substring(1);
                            if (SyntaxFactory.ParseToken(paramName).IsKeyword())
                            {
                                paramName = $"{paramName}_";
                            }
                            sb.Append($"\"{attr.Name}\", {paramName}.dtype, ");
                        }
                        else
                        {
                            string attrRealName = attr.Name;
                            if (SyntaxFactory.ParseToken(attrRealName).IsKeyword())
                            {
                                attrRealName = $"{attrRealName}_";
                            }
                            sb.Append($"\"{attr.Name}\", {attrRealName}, ");
                        }
                    }
                }
                else if(attr.Type == "int" && (op.InputArg.Any(x => x.NumberAttr == attr.Name) || op.OutputArg.Any(x => x.NumberAttr == attr.Name)))
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
            else if (outputArgsCount == 1)
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
            var attrValueDic = GetAttrsDefaultValue(op);
            foreach (var (key, (typeStr, _)) in attrValueDic)
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
            var attrValueDic = GetAttrsDefaultValue(op);
            foreach (var (key, _) in attrValueDic)
            {
                sb.Append($"keywords[\"{key}\"] = {key};");
            }
            sb.AppendLine($"var _op = tf.OpDefLib._apply_op_helper(\"{op.Name}\", name, keywords);");
        }

        // key, (type string, default value)
        public Dictionary<string, (string, string)> GetAttrsDefaultValue(OpDef op)
        {
            Dictionary<string, (string, string)> dic = new();
            foreach (var attr in op.Attr)
            {
                if (attr.Type == "type")
                {
                    bool found = op.InputArg.Any(x => x.TypeAttr == attr.Name);
                    if (!found)
                    {
                        if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.Type)
                        {
                            string name = Enum.GetName(typeof(TF_DataType), attr.DefaultValue.Type.as_tf_dtype());
                            string enumPath = typeof(TF_DataType).Name + "." + name;
                            dic[attr.Name] = ("TF_DataType", enumPath);
                        }
                        else
                        {
                            dic[attr.Name] = ("TF_DataType", "NOVALUE");
                        }
                    }
                }
                else if (attr.Type == "int")
                {
                    if(op.InputArg.Any(x => x.NumberAttr == attr.Name) || op.OutputArg.Any(x => x.NumberAttr == attr.Name))
                    {
                        continue;
                    }
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.I)
                    {
                        dic[attr.Name] = ("int", attr.DefaultValue.I.ToString());
                    }
                    else
                    {
                        dic[attr.Name] = ("int", "0");
                    }
                }
                else if (attr.Type == "float")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.F)
                    {
                        dic[attr.Name] = ("float", attr.DefaultValue.F.ToString() + "f");
                    }
                    else
                    {
                        dic[attr.Name] = ("float", "NOVALUE");
                    }
                }
                else if (attr.Type == "string")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.S)
                    {
                        dic[attr.Name] = ("string", $"\"{attr.DefaultValue.S.ToStringUtf8()}\"");
                    }
                    else
                    {
                        dic[attr.Name] = ("string", "NOVALUE");
                    }
                }
                else if (attr.Type == "bool")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.B)
                    {
                        dic[attr.Name] = ("bool", attr.DefaultValue.B.ToString().ToLower());
                    }
                    else
                    {
                        dic[attr.Name] = ("bool", "NOVALUE");
                    }
                }
                else if (attr.Type == "shape")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.Shape)
                    {
                        dic[attr.Name] = ("Shape", $"null");
                    }
                    else
                    {
                        dic[attr.Name] = ("Shape", "NOVALUE");
                    }
                }
                else if (attr.Type == "list(type)")
                {
                    dic[attr.Name] = ("TF_DataType[]", "NOVALUE");
                }
                else if (attr.Type == "list(shape)")
                {
                    dic[attr.Name] = ("Shape[]", "NOVALUE");
                }
                else if (attr.Type == "list(string)")
                {
                    dic[attr.Name] = ("string[]", "NOVALUE");
                }
                else if (attr.Type == "list(int)")
                {
                    dic[attr.Name] = ("int[]", "NOVALUE");
                }
                else if (attr.Type == "list(float)")
                {
                    dic[attr.Name] = ("float[]", "NOVALUE");
                }
                else if (attr.Type == "func")
                {
                    dic[attr.Name] = ("Func<Tensors, Tensors>", "NOVALUE");
                }
                else if (attr.Type == "list(func)")
                {
                    dic[attr.Name] = ("Func<Tensors, Tensors>[]", "NOVALUE");
                }
                else if (attr.Type == "tensor")
                {
                    dic[attr.Name] = ("TensorProto", "NOVALUE");
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            return dic;
        }

        private static bool HasRefArgs(OpDef op)
        {
            return op.InputArg.Any(x => x.IsRef);
        }
    }
}
