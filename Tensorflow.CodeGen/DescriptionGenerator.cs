using Microsoft.CodeAnalysis.CSharp;
using Protobuf.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Tensorflow.CodeGen
{
    public class DescriptionGenerator
    {
        private static readonly string replaceStrInner = "~~%~~";
        private static readonly string replaceStrInnerQuotationMarks = "^%^";
        Dictionary<string, Dictionary<string, string>> _opDescriptions = new Dictionary<string, Dictionary<string, string>>();
        Dictionary<string, OpDef> _opDescriptionDefs = new Dictionary<string, OpDef>();
        public DescriptionGenerator(string apiDefDirectory)
        {
            DirectoryInfo directory = new DirectoryInfo(apiDefDirectory);

            int errors = 0;
            foreach (FileInfo file in directory.GetFiles())
            {
                string target = file.Name.Split('.')[0].Split('_').Last();
                OpDef op = null;
                try
                {
                    op = ReadOpDefs(file.FullName).Op[0];
                }
                catch
                {
                    errors++;
                    continue;
                }
                _opDescriptionDefs[target] = op;
                _opDescriptions[target] = new Dictionary<string, string>();
                foreach (var arg in op.InputArg)
                {
                    string argName = arg.Name;
                    var token = SyntaxFactory.ParseToken(argName);
                    if (token.IsKeyword())
                    {
                        argName = $"{argName}_";
                    }
                    _opDescriptions[target][argName] = arg.Description ?? "";
                }
                foreach (var arg in op.Attr)
                {
                    var token = SyntaxFactory.ParseToken(arg.Name);
                    string realKey = arg.Name;
                    if (token.IsKeyword())
                    {
                        realKey += "_";
                    }
                    _opDescriptions[target][realKey] = arg.Description ?? "";
                }
                _opDescriptions[target]["SUMMARY"] = op.Summary ?? "";
                _opDescriptions[target]["DESC"] = op.Description ?? "";
            }
            Console.WriteLine($"Warning: {errors} description files cannot be analyzed! Please revise it if " +
                $"the failed files number is large, or ignore it.");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op"></param>
        /// <param name="sb"></param>
        public void AppendDescription(OpDef fullOp, StringBuilder sb)
        {
            var opName = fullOp.Name;
            if(_opDescriptions.TryGetValue(opName, out var op))
            {
                var def = _opDescriptionDefs[opName];
                sb.AppendLine("/// <summary>");
                sb.AppendLine($"/// {op["SUMMARY"]}");
                sb.AppendLine("/// </summary>");

                string totalDesc = op["DESC"];
                if (!string.IsNullOrEmpty(totalDesc))
                {
                    totalDesc = totalDesc.Replace(replaceStrInnerQuotationMarks, "\"");
                    sb.AppendLine("/// <remarks>");
                    string[] lines = totalDesc.Split(replaceStrInner);
                    foreach (var line in lines)
                    {
                        sb.AppendLine($"/// {line}");
                    }
                    sb.AppendLine("/// </remarks>");
                }

                var argNames = GetInputArgNames(fullOp);
                foreach (var argName in argNames)
                {
                    if(op.TryGetValue(argName, out var desc))
                    {
                        desc = desc.Replace(replaceStrInnerQuotationMarks, "\"");
                        string[] lines = desc.Split(replaceStrInner);
                        sb.AppendLine($"/// <param name=\"{argName}\">");
                        foreach (var line in lines)
                        {
                            sb.AppendLine($"/// {line}");
                        }
                        sb.AppendLine("/// </param>");
                    }
                    else
                    {
                        sb.AppendLine($"/// <param name=\"{argName}\"></param>");
                    }
                }

                List<string> returnValueDescs = new();
                foreach (var arg in def.OutputArg)
                {
                    if (!string.IsNullOrEmpty(arg.Description))
                    {
                        returnValueDescs.Add($"{arg.Name}: {arg.Description}");
                    }
                }
                string returnValueDesc = "";
                if (returnValueDescs.Count > 0)
                {
                    returnValueDesc = string.Join(" && ", returnValueDescs);
                }
                sb.AppendLine($"/// <returns>{returnValueDesc}</returns>");
            }
            else
            {
                sb.AppendLine("/// <summary>");
                sb.AppendLine($"///");
                sb.AppendLine("/// </summary>");

                var argNames = GetInputArgNames(fullOp);
                foreach (var argName in argNames)
                {
                    sb.AppendLine($"/// <param name=\"{argName}\"></param>");
                }

                sb.AppendLine($"/// <returns></returns>");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">
        /// </param>
        /// <returns></returns>
        /// <remarks></remarks>
        public List<string> GetInputArgNames(OpDef op)
        {
            List<string> names = new();
            foreach (var arg in op.InputArg)
            {
                string argName = arg.Name;
                var token = SyntaxFactory.ParseToken(argName);
                if (token.IsKeyword())
                {
                    argName = $"{argName}_";
                }
                names.Add(argName);
            }
            var attrValueDic = Utils.GetAttrsDefaultValue(op, out var dynamicDefaultValues);
            foreach (var (key, typeStr, value) in attrValueDic)
            {
                var token = SyntaxFactory.ParseToken(key);
                string realKey = key;
                if (token.IsKeyword())
                {
                    realKey += "_";
                }
                names.Add(realKey);
            }
            return names;
        }

        private static OpList ReadOpDefs(string path)
        {
            var text = File.ReadAllText(path);
            text = RemoveLintTags(text);
            text = PreProcessText(text);

            string pattern = @"<<END([\s\S]*?)END";

            // 定义用于替换的字符串
            string replaceStrPrefix = "\"";
            string replaceStrSuffix = "\"";

            // 将匹配到的文本段全部替换
            string replacedText = Regex.Replace(text, pattern, match => {
                string matchedText = match.Value;
                string innerText = match.Groups[1].Value;
                innerText = innerText.Replace("\"", replaceStrInnerQuotationMarks)
                        .Replace("\r\n", replaceStrInner).Replace("\n", replaceStrInner); // 替换内部换行符
                return replaceStrPrefix + innerText + replaceStrSuffix; // 替换首尾
            }, RegexOptions.Multiline);

            var opDefs = new TextParser(TextParser.Settings.Default.WithIgnoreUnknownFields(true)).Parse<OpList>(replacedText);
            return opDefs;
        }

        static string PreProcessText(string input)
        {
            int depth = 0;
            int endBlockDepth = -1;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.Length; i++)
            {
                char c = input[i];
                if (c == '{')
                {
                    depth++;
                    sb.Append(c);
                }
                else if (c == '}')
                {
                    if (depth == endBlockDepth)
                    {
                        sb.Append("END\n");
                        endBlockDepth = -1;
                    }
                    sb.Append(c);
                    depth--;
                }
                else if (c == '<' && i + 5 < input.Length && input.Substring(i, 5) == "<<END")
                {
                    endBlockDepth = depth;
                    sb.Append("<<END");
                    i += 4;
                }
                else if (c == 'E' && i + 3 < input.Length && input.Substring(i, 3) == "END")
                {
                    endBlockDepth = -1;
                    sb.Append("END");
                    i += 2;
                }
                else
                {
                    sb.Append(c);
                }
            }

            string output = sb.ToString();
            return output;
        }

        static string RemoveLintTags(string input)
        {
            string[] lines = input.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);
            StringBuilder sb = new StringBuilder();
            foreach (string line in lines)
            {
                if (!line.TrimStart().StartsWith("# LINT"))
                {
                    sb.AppendLine(line);
                }
            }
            return sb.ToString().TrimEnd();
        }
    }
}
