using System.Collections.Generic;
using System.Linq;
using System;
using static Tensorflow.OpDef.Types;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class wrap_tfe_src
    {
        public static void SetOpAttrs(Context ctx, TFE_Op op, object[] attrs, Status out_status)
        {
            var len = attrs.Length;
            for (int i = 0; i < len; i += 2)
            {
                var key = attrs[i].ToString();
                var value = attrs[i + 1];

                byte is_list = 0; 
                var type = c_api.TFE_OpGetAttrType(op, key, ref is_list, out_status);
                if (!out_status.ok()) return;
                if (is_list != 0)
                    SetOpAttrList(ctx, op, key, value, type, null, out_status);
                else
                    SetOpAttrScalar(ctx, op, key, value, type, null, out_status);
                out_status.Check(true);
            }
        }

        /// <summary>
        /// This function will set the op attrs required. If an attr has the value of
        /// None, then it will read the AttrDef to get the default value and set that
        /// instead. Any failure in this function will simply fall back to the slow
        /// path.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="op"></param>
        /// <param name="attr"></param>
        /// <param name="attr_name"></param>
        /// <param name="attr_value"></param>
        /// <param name="attr_list_sizes"></param>
        /// <param name="status"></param>
        private static void SetOpAttrWithDefaults(Context ctx, IntPtr op, AttrDef attr, 
            string attr_name, object attr_value,  
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            byte is_list = 0;
            var type = c_api.TFE_OpGetAttrType(op, attr_name, ref is_list, status);
            if (status.Code != TF_Code.TF_OK) return;

            if(attr_value == null)
            {
                if (is_list != 0)
                    ;
                //SetOpAttrListDefault
                else
                    ;
                //SetOpAttrScalarDefault
            }
            else
            {
                if (is_list != 0)
                    ;//  SetOpAttrList
                else
                    SetOpAttrScalar(ctx, op, attr_name, attr_value, type, attr_list_sizes, status);
            }
        }

        private static bool SetOpAttrList(Context ctx, IntPtr op,
            string key, object value, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            return false;
        }

        private static bool SetOpAttrScalar(Context ctx, IntPtr op, 
            string key, object value, TF_AttrType type,
            Dictionary<string, long> attr_list_sizes,
            Status status)
        {
            switch(type)
            {
                case TF_AttrType.TF_ATTR_STRING:
                    c_api.TFE_OpSetAttrString(op, key, value.ToString(), (uint)value.ToString().Length);
                    break;
                case TF_AttrType.TF_ATTR_TYPE:
                    c_api.TFE_OpSetAttrType(op, key, (TF_DataType)value);
                    break;
                case TF_AttrType.TF_ATTR_BOOL:
                    c_api.TFE_OpSetAttrBool(op, key, Convert.ToBoolean(value));
                    break;
                case TF_AttrType.TF_ATTR_INT:
                    c_api.TFE_OpSetAttrInt(op, key, Convert.ToInt64(value));
                    break;
                case TF_AttrType.TF_ATTR_SHAPE:
                    var dims = (value as int[]).Select(x => (long)x).ToArray();
                    c_api.TFE_OpSetAttrShape(op, key, dims, dims.Length, status);
                    status.Check(true);
                    break;
                default:
                    throw new NotImplementedException($"SetOpAttrScalar for {type}");
            }

            return true;
        }
    }
}
