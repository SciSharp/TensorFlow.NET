// Record the gradient for a given op.
extern void TFE_Py_RecordGradient(const char* op_name, void* inputs,
                                void* attrs, void* results,
                                const char* name);