using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class GarbageCollector
    {
        static Dictionary<IntPtr, GCItemCounter> container = new Dictionary<IntPtr, GCItemCounter>();

        static object locker = new object();
        public static void Init()
        {
            Task.Run(() =>
            {
                while (true)
                {
                    Thread.Sleep(100);
                    Recycle();
                }
            });

        }

        public static void Increase(IntPtr handle, GCItemType type)
        {
            if (handle == IntPtr.Zero)
                return;

            if (container.ContainsKey(handle))
            {
                container[handle].RefCounter++;
                container[handle].LastUpdateTime = DateTime.Now;
            }
            else
            {
                lock (locker)
                {
                    container[handle] = new GCItemCounter
                    {
                        ItemType = type,
                        RefCounter = 1,
                        Handle = handle,
                        LastUpdateTime = DateTime.Now
                    };
                }
            }
        }

        public static void Decrease(IntPtr handle)
        {
            lock (locker)
            {
                if (handle != IntPtr.Zero && container.ContainsKey(handle))
                    container[handle].RefCounter--;
            }
        }

        private static void Recycle()
        {
            // dispose before 1 sec
            lock (locker)
            {
                var items = container.Values
                    .Where(x => x.RefCounter <= 0 && (DateTime.Now - x.LastUpdateTime).TotalMilliseconds > 300)
                    .ToArray();

                foreach (var item in items)
                {
                    item.RefCounter = 0;
                    container.Remove(item.Handle);
                    switch (item.ItemType)
                    {
                        case GCItemType.TensorHandle:
                            //print($"c_api.TF_DeleteTensor({item.Handle.ToString("x16")})");
                            c_api.TF_DeleteTensor(item.Handle);
                            break;
                        case GCItemType.LocalTensorHandle:
                            //print($"c_api.TFE_DeleteTensorHandle({item.Handle.ToString("x16")})");
                            c_api.TFE_DeleteTensorHandle(item.Handle);
                            break;
                        case GCItemType.EagerTensorHandle:
                            //print($"c_api.TFE_DeleteEagerTensor({item.Handle.ToString("x16")})");
                            c_api.TFE_DeleteEagerTensor(item.Handle);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
}
