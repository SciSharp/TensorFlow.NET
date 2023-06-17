using Protobuf.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow.CodeGen
{
    public static class Utils
    {
        public static string ConvertToUnderscore(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }

            StringBuilder result = new StringBuilder();

            int state = 1; // the previous char was not lowered.
            for (int i = 0; i < input.Length; i++)
            {
                char current = input[i];

                // 首字母不需要添加下划线
                if (char.IsUpper(current))
                {
                    if(i > 0)
                    {
                        char pre = input[i - 1];
                        if (char.IsDigit(pre))
                        {
                            result.Append(char.ToLower(current));
                            continue;
                        }
                    }
                    if (state == 0)
                    {
                        result.Append("_");
                        state = 1;
                    }
                    result.Append(char.ToLower(current));
                }
                else
                {
                    result.Append(char.ToLower(current));
                    state = 0;
                }
            }

            return result.ToString();
        }

        public static OpList ReadAllOpDefs(string path)
        {
            var text = File.ReadAllText(path);
            var opDefs = OpList.Parser.ParseText(text);
            return opDefs;
        }

        // name, type string, default value
        public static List<(string, string, string)> GetAttrsDefaultValue(OpDef op, out Dictionary<string, string> dynamicDefaultValues)
        {
            dynamicDefaultValues = new();
            List<(string, string, string)> res = new();
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
                            res.Add((attr.Name, "TF_DataType", enumPath));
                        }
                        else
                        {
                            res.Add((attr.Name, "TF_DataType", "NOVALUE"));
                        }
                    }
                }
                else if (attr.Type == "int")
                {
                    if (op.InputArg.Any(x => x.NumberAttr == attr.Name))
                    {
                        continue;
                    }
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.I)
                    {
                        res.Add((attr.Name, "int", attr.DefaultValue.I.ToString()));
                    }
                    else
                    {
                        res.Add((attr.Name, "int", "0"));
                    }
                }
                else if (attr.Type == "float")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.F)
                    {
                        res.Add((attr.Name, "float", attr.DefaultValue.F.ToString() + "f"));
                    }
                    else
                    {
                        res.Add((attr.Name, "float", "NOVALUE"));
                    }
                }
                else if (attr.Type == "string")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.S)
                    {
                        res.Add((attr.Name, "string", $"\"{attr.DefaultValue.S.ToStringUtf8()}\""));
                    }
                    else
                    {
                        res.Add((attr.Name, "string", "NOVALUE"));
                    }
                }
                else if (attr.Type == "bool")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.B)
                    {
                        res.Add((attr.Name, "bool", attr.DefaultValue.B.ToString().ToLower()));
                    }
                    else
                    {
                        res.Add((attr.Name, "bool", "NOVALUE"));
                    }
                }
                else if (attr.Type == "shape")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.Shape)
                    {
                        if (attr.DefaultValue.Shape.UnknownRank)
                        {
                            res.Add((attr.Name, "Shape", $"null"));
                        }
                        else
                        {
                            Shape shape = new Shape(attr.DefaultValue.Shape);
                            string expression = $"new Shape({string.Join(", ", shape.dims)})";
                            dynamicDefaultValues[attr.Name] = expression;
                            res.Add((attr.Name, "Shape", $"null"));
                        }
                    }
                    else
                    {
                        res.Add((attr.Name, "Shape", "NOVALUE"));
                    }
                }
                else if (attr.Type == "list(type)")
                {
                    if(op.InputArg.Any(x => x.TypeListAttr == attr.Name))
                    {
                        continue;
                    }
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.Type)
                    {
                        List<TF_DataType> values = new();
                        foreach (var value in attr.DefaultValue.List.Type)
                        {
                            values.Add(value.as_tf_dtype());
                        }
                        string expression = "new TF_DataType[]{" + $"{string.Join(", ", values)}" + "}";
                        dynamicDefaultValues[attr.Name] = expression;
                        res.Add((attr.Name, "TF_DataType[]", $"null"));
                    }
                    else
                    {
                        res.Add((attr.Name, "TF_DataType[]", "NOVALUE"));
                    }
                }
                else if (attr.Type == "list(shape)")
                {
                    res.Add((attr.Name, "Shape[]", "NOVALUE"));
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.List)
                    {
                        List<string> exps = new();
                        foreach (var value in attr.DefaultValue.List.Shape)
                        {
                            exps.Add($"new Shape({string.Join(", ", value.Dim.Select(x => x.Size))})");
                        }
                        string expression = "new Shape[]{" + $"{string.Join(", ", exps)}" + "}";
                        dynamicDefaultValues[attr.Name] = expression;
                        res.Add((attr.Name, "string[]", $"null"));
                    }
                    else
                    {
                        res.Add((attr.Name, "string[]", "NOVALUE"));
                    }
                }
                else if (attr.Type == "list(string)")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.List)
                    {
                        List<string> values = new();
                        foreach (var value in attr.DefaultValue.List.S)
                        {
                            values.Add(value.ToStringUtf8());
                        }
                        string expression = "new string[]{" + $"{string.Join(", ", values)}" + "}";
                        dynamicDefaultValues[attr.Name] = expression;
                        res.Add((attr.Name, "string[]", $"null"));
                    }
                    else
                    {
                        res.Add((attr.Name, "string[]", "NOVALUE"));
                    }
                }
                else if (attr.Type == "list(int)")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.List)
                    {
                        List<int> values = new();
                        foreach (var value in attr.DefaultValue.List.I)
                        {
                            values.Add((int)value);
                        }
                        string expression = "new int[]{" + $"{string.Join(", ", values)}" + "}";
                        dynamicDefaultValues[attr.Name] = expression;
                        res.Add((attr.Name, "int[]", $"null"));
                    }
                    else
                    {
                        res.Add((attr.Name, "int[]", "NOVALUE"));
                    }
                }
                else if (attr.Type == "list(float)")
                {
                    if (attr.DefaultValue is not null && attr.DefaultValue.ValueCase == AttrValue.ValueOneofCase.List)
                    {
                        List<float> values = new();
                        foreach (var value in attr.DefaultValue.List.F)
                        {
                            values.Add(value);
                        }
                        string expression = "new float[]{" + $"{string.Join(", ", values)}" + "}";
                        dynamicDefaultValues[attr.Name] = expression;
                        res.Add((attr.Name, "float[]", $"null"));
                    }
                    else
                    {
                        res.Add((attr.Name, "float[]", "NOVALUE"));
                    }
                }
                else if (attr.Type == "func")
                {
                    res.Add((attr.Name, "object", "NOVALUE"));
                }
                else if (attr.Type == "list(func)")
                {
                    res.Add((attr.Name, "object[]", "NOVALUE"));
                }
                else if (attr.Type == "tensor")
                {
                    res.Add((attr.Name, "TensorProto", "NOVALUE"));
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            return res;
        }
    }
}
