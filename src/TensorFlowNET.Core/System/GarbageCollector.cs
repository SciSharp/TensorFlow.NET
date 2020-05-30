using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Timers;

namespace Tensorflow
{
    public class GarbageCollector
    {
        static Dictionary<IntPtr, GCItemCounter> container = new Dictionary<IntPtr, GCItemCounter>();
        static Timer timer = null;
        static object locker = new object();

        public static void Increase(IntPtr handle, GCItemType type)
        {
            if(timer == null)
            {
                timer = new Timer(300);
                // Hook up the Elapsed event for the timer. 
                timer.Elapsed += OnTimedEvent;
                timer.AutoReset = true;
                timer.Enabled = true;
            }

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
                if (container.ContainsKey(handle))
                    container[handle].RefCounter--;
            }
        }

        private static void OnTimedEvent(object source, ElapsedEventArgs e)
        {
            timer.Stop();

            // dispose before 1 sec
            lock (locker)
            {
                var items = container.Values
                    .Where(x => x.RefCounter <= 0 && (DateTime.Now - x.LastUpdateTime).Milliseconds > 300)
                    .ToArray();

                foreach (var item in items)
                {
                    item.RefCounter = 0;
                    container.Remove(item.Handle);
                    switch (item.ItemType)
                    {
                        case GCItemType.TensorHandle:
                            c_api.TF_DeleteTensor(item.Handle);
                            break;
                        case GCItemType.LocalTensorHandle:
                            c_api.TFE_DeleteTensorHandle(item.Handle);
                            break;
                        case GCItemType.EagerTensorHandle:
                            c_api.TFE_DeleteEagerTensor(item.Handle);
                            break;
                        default:
                            break;
                    }
                }
            }

            timer.Start();
        }
    }
}
