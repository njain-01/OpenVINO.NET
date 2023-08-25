using System;
using System.Runtime.InteropServices;

namespace Sdcb.OpenVINO.Natives;

public static unsafe partial class NativeMethods
{
    /// <summary>Print the error info.</summary>
    /// <param name="ov_status_e">a status code.</param>
    [DllImport(Dll)] public static extern byte* ov_get_error_info(ov_status_e status);
    
    /// <summary>free char</summary>
    /// <param name="content">The pointer to the char to free.</param>
    [DllImport(Dll)] public static extern void ov_free(byte* content);
    
    /// <summary>Check this dimension whether is dynamic</summary>
    /// <param name="dim">The dimension pointer that will be checked.</param>
    [DllImport(Dll)] public static extern bool ov_dimension_is_dynamic(ov_dimension dim);
    
    /// <summary>Create a layout object.</summary>
    /// <param name="layout">The layout input pointer.</param>
    /// <param name="layout_desc">The description of layout.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_layout_create(byte* layout_desc, ov_layout** layout);
    
    /// <summary>Free layout object.</summary>
    /// <param name="layout">will be released.</param>
    [DllImport(Dll)] public static extern void ov_layout_free(ov_layout* layout);
    
    /// <summary>Convert layout object to a readable string.</summary>
    /// <param name="layout">will be converted.</param>
    [DllImport(Dll)] public static extern byte* ov_layout_to_string(ov_layout* layout);
    
    /// <summary>Check this rank whether is dynamic</summary>
    /// <param name="rank">The rank pointer that will be checked.</param>
    [DllImport(Dll)] public static extern bool ov_rank_is_dynamic(ov_dimension rank);
    
    /// <summary>Initialize a fully shape object, allocate space for its dimensions and set its content id dims is not null.</summary>
    /// <param name="rank">The rank value for this object, it should be more than 0(>0)</param>
    /// <param name="dims">The dimensions data for this shape object, it's size should be equal to rank.</param>
    /// <param name="shape">The input/output shape object pointer.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_shape_create(long rank, long* dims, ov_shape_t* shape);
    
    /// <summary>Free a shape object's internal memory.</summary>
    /// <param name="shape">The input shape object pointer.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_shape_free(ov_shape_t* shape);
    
    /// <summary>Initialze a partial shape with static rank and dynamic dimension.</summary>
    /// <param name="rank">support static rank.</param>
    /// <param name="dims">support dynamic and static dimension.   Static rank, but dynamic dimensions on some or all axes.      Examples: `{1,2,?,4}` or `{?,?,?}` or `{1,2,-1,4}`   Static rank, and static dimensions on all axes.      Examples: `{1,2,3,4}` or `{6}` or `{}`</param>
    [DllImport(Dll)] public static extern ov_status_e ov_partial_shape_create(long rank, ov_dimension* dims, ov_partial_shape* partial_shape_obj);
    
    /// <summary>Initialze a partial shape with dynamic rank and dynamic dimension.</summary>
    /// <param name="rank">support dynamic and static rank.</param>
    /// <param name="dims">support dynamic and static dimension.   Dynamic rank:      Example: `?`   Static rank, but dynamic dimensions on some or all axes.      Examples: `{1,2,?,4}` or `{?,?,?}` or `{1,2,-1,4}`   Static rank, and static dimensions on all axes.      Examples: `{1,2,3,4}` or `{6}` or `{}"`</param>
    [DllImport(Dll)] public static extern ov_status_e ov_partial_shape_create_dynamic(ov_dimension rank, ov_dimension* dims, ov_partial_shape* partial_shape_obj);
    
    /// <summary>Initialize a partial shape with static rank and static dimension.</summary>
    /// <param name="rank">support static rank.</param>
    /// <param name="dims">support static dimension.   Static rank, and static dimensions on all axes.      Examples: `{1,2,3,4}` or `{6}` or `{}`</param>
    [DllImport(Dll)] public static extern ov_status_e ov_partial_shape_create_static(long rank, long* dims, ov_partial_shape* partial_shape_obj);
    
    /// <summary>Release internal memory allocated in partial shape.</summary>
    /// <param name="partial_shape">The object's internal memory will be released.</param>
    [DllImport(Dll)] public static extern void ov_partial_shape_free(ov_partial_shape* partial_shape);
    
    /// <summary>Convert partial shape without dynamic data to a static shape.</summary>
    /// <param name="partial_shape">The partial_shape pointer.</param>
    /// <param name="shape">The shape pointer.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_partial_shape_to_shape(ov_partial_shape partial_shape, ov_shape_t* shape);
    
    /// <summary>Convert shape to partial shape.</summary>
    /// <param name="shape">The shape pointer.</param>
    /// <param name="partial_shape">The partial_shape pointer.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_shape_to_partial_shape(ov_shape_t shape, ov_partial_shape* partial_shape);
    
    /// <summary>Check this partial_shape whether is dynamic</summary>
    /// <param name="partial_shape">The partial_shape pointer.</param>
    [DllImport(Dll)] public static extern bool ov_partial_shape_is_dynamic(ov_partial_shape partial_shape);
    
    /// <summary>Helper function, convert a partial shape to readable string.</summary>
    /// <param name="partial_shape">The partial_shape pointer.</param>
    [DllImport(Dll)] public static extern byte* ov_partial_shape_to_string(ov_partial_shape partial_shape);
    
    /// <summary>Get the shape of port object.</summary>
    /// <param name="port">A pointer to ov_output_const_port_t.</param>
    /// <param name="tensor_shape">tensor shape.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_const_port_get_shape(ov_output_const_port* port, ov_shape_t* tensor_shape);
    
    /// <summary>Get the shape of port object.</summary>
    /// <param name="port">A pointer to ov_output_port_t.</param>
    /// <param name="tensor_shape">tensor shape.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_port_get_shape(ov_output_port* port, ov_shape_t* tensor_shape);
    
    /// <summary>Get the tensor name of port.</summary>
    /// <param name="port">A pointer to the ov_output_const_port_t.</param>
    /// <param name="tensor_name">A pointer to the tensor name.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_port_get_any_name(ov_output_const_port* port, byte** tensor_name);
    
    /// <summary>Get the partial shape of port.</summary>
    /// <param name="port">A pointer to the ov_output_const_port_t.</param>
    /// <param name="partial_shape">Partial shape.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_port_get_partial_shape(ov_output_const_port* port, ov_partial_shape* partial_shape);
    
    /// <summary>Get the tensor type of port.</summary>
    /// <param name="port">A pointer to the ov_output_const_port_t.</param>
    /// <param name="tensor_type">tensor type.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_port_get_element_type(ov_output_const_port* port, ov_element_type_e* tensor_type);
    
    /// <summary>free port object</summary>
    /// <param name="port">The pointer to the instance of the ov_output_port_t to free.</param>
    [DllImport(Dll)] public static extern void ov_output_port_free(ov_output_port* port);
    
    /// <summary>free const port</summary>
    /// <param name="port">The pointer to the instance of the ov_output_const_port_t to free.</param>
    [DllImport(Dll)] public static extern void ov_output_const_port_free(ov_output_const_port* port);
    
    /// <summary>Constructs Tensor using element type and shape. Allocate internal host storage using default allocator</summary>
    /// <param name="type">Tensor element type</param>
    /// <param name="shape">Tensor shape</param>
    /// <param name="host_ptr">Pointer to pre-allocated host memory</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_create_from_host_ptr(ov_element_type_e type, ov_shape_t shape, void* host_ptr, ov_tensor** tensor);
    
    /// <summary>Constructs Tensor using element type and shape. Allocate internal host storage using default allocator</summary>
    /// <param name="type">Tensor element type</param>
    /// <param name="shape">Tensor shape</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_create(ov_element_type_e type, ov_shape_t shape, ov_tensor** tensor);
    
    /// <summary>Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.</summary>
    /// <param name="shape">Tensor shape</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_set_shape(ov_tensor* tensor, ov_shape_t shape);
    
    /// <summary>Get shape for tensor.</summary>
    /// <param name="shape">Tensor shape</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_get_shape(ov_tensor* tensor, ov_shape_t* shape);
    
    /// <summary>Get type for tensor.</summary>
    /// <param name="type">Tensor element type</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_get_element_type(ov_tensor* tensor, ov_element_type_e* type);
    
    /// <summary>the total number of elements (a product of all the dims or 1 for scalar).</summary>
    /// <param name="elements_size">number of elements</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_get_size(ov_tensor* tensor, ulong* elements_size);
    
    /// <summary>the size of the current Tensor in bytes.</summary>
    /// <param name="byte_size">the size of the current Tensor in bytes.</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_get_byte_size(ov_tensor* tensor, ulong* byte_size);
    
    /// <summary>Provides an access to the underlaying host memory.</summary>
    /// <param name="data">A point to host memory.</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_tensor_data(ov_tensor* tensor, void** data);
    
    /// <summary>Free ov_tensor_t.</summary>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern void ov_tensor_free(ov_tensor* tensor);
    
    /// <summary>Set an input/output tensor to infer on by the name of tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="tensor_name">Name of the input or output tensor.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_tensor(ov_infer_request* infer_request, byte* tensor_name, ov_tensor* tensor);
    
    /// <summary>Set an input/output tensor to infer request for the port.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="port">Port of the input or output tensor, which can be got by calling ov_model_t/ov_compiled_model_t interface.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_tensor_by_port(ov_infer_request* infer_request, ov_output_port* port, ov_tensor* tensor);
    
    /// <summary>Set an input/output tensor to infer request for the port.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="port">Const port of the input or output tensor, which can be got by call interface from  ov_model_t/ov_compiled_model_t.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_tensor_by_const_port(ov_infer_request* infer_request, ov_output_const_port* port, ov_tensor* tensor);
    
    /// <summary>Set an input tensor to infer on by the index of tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="idx">Index of the input port. If   is greater than the number of model inputs, an error will return.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_input_tensor_by_index(ov_infer_request* infer_request, ulong idx, ov_tensor* tensor);
    
    /// <summary>Set an input tensor for the model with single input to infer on.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_input_tensor(ov_infer_request* infer_request, ov_tensor* tensor);
    
    /// <summary>Set an output tensor to infer by the index of output tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="idx">Index of the output tensor.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_output_tensor_by_index(ov_infer_request* infer_request, ulong idx, ov_tensor* tensor);
    
    /// <summary>Set an output tensor to infer models with single output.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_output_tensor(ov_infer_request* infer_request, ov_tensor* tensor);
    
    /// <summary>Get an input/output tensor by the name of tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="tensor_name">Name of the input or output tensor to get.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_tensor(ov_infer_request* infer_request, byte* tensor_name, ov_tensor** tensor);
    
    /// <summary>Get an input/output tensor by const port.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="port">Port of the tensor to get.   is not found, an error will return.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_tensor_by_const_port(ov_infer_request* infer_request, ov_output_const_port* port, ov_tensor** tensor);
    
    /// <summary>Get an input/output tensor by port.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="port">Port of the tensor to get.   is not found, an error will return.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_tensor_by_port(ov_infer_request* infer_request, ov_output_port* port, ov_tensor** tensor);
    
    /// <summary>Get an input tensor by the index of input tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="idx">Index of the tensor to get.   If the tensor with the specified   is not found, an error will  return.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_input_tensor_by_index(ov_infer_request* infer_request, ulong idx, ov_tensor** tensor);
    
    /// <summary>Get an input tensor from the model with only one input tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_input_tensor(ov_infer_request* infer_request, ov_tensor** tensor);
    
    /// <summary>Get an output tensor by the index of output tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="idx">Index of the tensor to get.   If the tensor with the specified   is not found, an error will  return.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_output_tensor_by_index(ov_infer_request* infer_request, ulong idx, ov_tensor** tensor);
    
    /// <summary>Get an output tensor from the model with only one output tensor.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="tensor">Reference to the tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_output_tensor(ov_infer_request* infer_request, ov_tensor** tensor);
    
    /// <summary>Infer specified input(s) in synchronous mode.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_infer(ov_infer_request* infer_request);
    
    /// <summary>Cancel inference request.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_cancel(ov_infer_request* infer_request);
    
    /// <summary>Start inference of specified input(s) in asynchronous mode.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_start_async(ov_infer_request* infer_request);
    
    /// <summary>Wait for the result to become available.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_wait(ov_infer_request* infer_request);
    
    /// <summary>Waits for the result to become available. Blocks until the specified timeout has elapsed or the result becomes available, whichever comes first.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="timeout">Maximum duration, in milliseconds, to block for.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_wait_for(ov_infer_request* infer_request, long timeout);
    
    /// <summary>Set callback function, which will be called when inference is done.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="callback">A function to be called.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_set_callback(ov_infer_request* infer_request, ov_callback_t* callback);
    
    /// <summary>Release the memory allocated by ov_infer_request_t.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_infer_request_free(ov_infer_request* infer_request);
    
    /// <summary>Query performance measures per layer to identify the most time consuming operation.</summary>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    /// <param name="profiling_infos">Vector of profiling information for operations in a model.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_infer_request_get_profiling_info(ov_infer_request* infer_request, ov_profiling_info_list_t* profiling_infos);
    
    /// <summary>Release the memory allocated by ov_profiling_info_list_t.</summary>
    /// <param name="profiling_infos">A pointer to the ov_profiling_info_list_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_profiling_info_list_free(ov_profiling_info_list_t* profiling_infos);
    
    /// <summary>Release the memory allocated by ov_model_t.</summary>
    /// <param name="model">A pointer to the ov_model_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_model_free(ov_model* model);
    
    /// <summary>Get a const input port of ov_model_t,which only support single input model.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="input_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_const_input(ov_model* model, ov_output_const_port** input_port);
    
    /// <summary>Get a const input port of ov_model_t by name.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="tensor_name">The name of input tensor.</param>
    /// <param name="input_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_const_input_by_name(ov_model* model, byte* tensor_name, ov_output_const_port** input_port);
    
    /// <summary>Get a const input port of ov_model_t by port index.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="index">input tensor index.</param>
    /// <param name="input_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_const_input_by_index(ov_model* model, ulong index, ov_output_const_port** input_port);
    
    /// <summary>Get single input port of ov_model_t, which only support single input model.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="input_port">A pointer to the ov_output_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_input(ov_model* model, ov_output_port** input_port);
    
    /// <summary>Get an input port of ov_model_t by name.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="tensor_name">input tensor name (char *).</param>
    /// <param name="input_port">A pointer to the ov_output_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_input_by_name(ov_model* model, byte* tensor_name, ov_output_port** input_port);
    
    /// <summary>Get an input port of ov_model_t by port index.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="index">input tensor index.</param>
    /// <param name="input_port">A pointer to the ov_output_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_input_by_index(ov_model* model, ulong index, ov_output_port** input_port);
    
    /// <summary>Get a single const output port of ov_model_t, which only support single output model..</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_const_output(ov_model* model, ov_output_const_port** output_port);
    
    /// <summary>Get a const output port of ov_model_t by port index.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="index">input tensor index.</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_const_output_by_index(ov_model* model, ulong index, ov_output_const_port** output_port);
    
    /// <summary>Get a const output port of ov_model_t by name.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="tensor_name">input tensor name (char *).</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_const_output_by_name(ov_model* model, byte* tensor_name, ov_output_const_port** output_port);
    
    /// <summary>Get an single output port of ov_model_t, which only support single output model.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_output(ov_model* model, ov_output_port** output_port);
    
    /// <summary>Get an output port of ov_model_t by port index.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="index">input tensor index.</param>
    /// <param name="output_port">A pointer to the ov_output_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_output_by_index(ov_model* model, ulong index, ov_output_port** output_port);
    
    /// <summary>Get an output port of ov_model_t by name.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="tensor_name">output tensor name (char *).</param>
    /// <param name="output_port">A pointer to the ov_output_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_output_by_name(ov_model* model, byte* tensor_name, ov_output_port** output_port);
    
    /// <summary>Get the input size of ov_model_t.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="input_size">the model's input size.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_inputs_size(ov_model* model, ulong* input_size);
    
    /// <summary>Get the output size of ov_model_t.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="output_size">the model's output size.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_outputs_size(ov_model* model, ulong* output_size);
    
    /// <summary>Returns true if any of the ops defined in the model is dynamic shape..</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    [DllImport(Dll)] public static extern bool ov_model_is_dynamic(ov_model* model);
    
    /// <summary>Do reshape in model with a list of <name, partial shape>.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="tensor_names">The list of input tensor names.</param>
    /// <param name="partialShape">A PartialShape list.</param>
    /// <param name="size">The item count in the list.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_reshape(ov_model* model, byte** tensor_names, ov_partial_shape* partial_shapes, ulong size);
    
    /// <summary>Do reshape in model with partial shape for a specified name.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="tensor_name">The tensor name of input tensor.</param>
    /// <param name="partialShape">A PartialShape.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_reshape_input_by_name(ov_model* model, byte* tensor_name, ov_partial_shape partial_shape);
    
    /// <summary>Do reshape in model for one node(port 0).</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="partialShape">A PartialShape.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_reshape_single_input(ov_model* model, ov_partial_shape partial_shape);
    
    /// <summary>Do reshape in model with a list of <port id, partial shape>.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="port_indexes">The array of port indexes.</param>
    /// <param name="partialShape">A PartialShape list.</param>
    /// <param name="size">The item count in the list.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_reshape_by_port_indexes(ov_model* model, ulong* port_indexes, ov_partial_shape* partial_shape, ulong size);
    
    /// <summary>Do reshape in model with a list of <ov_output_port_t, partial shape>.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="output_ports">The ov_output_port_t list.</param>
    /// <param name="partialShape">A PartialShape list.</param>
    /// <param name="size">The item count in the list.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_reshape_by_ports(ov_model* model, ov_output_port** output_ports, ov_partial_shape* partial_shapes, ulong size);
    
    /// <summary>Gets the friendly name for a model.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="friendly_name">the model's friendly name.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_model_get_friendly_name(ov_model* model, byte** friendly_name);
    
    /// <summary>Allocates memory tensor in device memory or wraps user-supplied memory handle using the specified tensor description and low-level device-specific parameters. Returns a pointer to the object that implements the RemoteTensor interface.</summary>
    /// <param name="context">A pointer to the ov_remote_context_t instance.</param>
    /// <param name="type">Defines the element type of the tensor.</param>
    /// <param name="shape">Defines the shape of the tensor.</param>
    /// <param name="object_args_size">Size of the low-level tensor object parameters.</param>
    /// <param name="remote_tensor">Pointer to returned ov_tensor_t that contains remote tensor instance.</param>
    /// <param name="variadic">params Contains low-level tensor object parameters.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_remote_context_create_tensor(ov_remote_context* context, ov_element_type_e type, ov_shape_t shape, ulong object_args_size, ov_tensor** remote_tensor);
    
    /// <summary>Returns name of a device on which underlying object is allocated.</summary>
    /// <param name="context">A pointer to the ov_remote_context_t instance.</param>
    /// <param name="device_name">Device name will be returned.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_remote_context_get_device_name(ov_remote_context* context, byte** device_name);
    
    /// <summary>Returns a string contains device-specific parameters required for low-level operations with the underlying object. Parameters include device/context handles, access flags, etc. Content of the returned map depends on a remote execution context that is currently set on the device (working scenario). One actaul example: "CONTEXT_TYPE OCL OCL_CONTEXT 0x5583b2ec7b40 OCL_QUEUE 0x5583b2e98ff0"</summary>
    /// <param name="context">A pointer to the ov_remote_context_t instance.</param>
    /// <param name="size">The size of param pairs.</param>
    /// <param name="params">Param name:value list.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_remote_context_get_params(ov_remote_context* context, ulong* size, byte** @params);
    
    /// <summary>This method is used to create a host tensor object friendly for the device in current context. For example, GPU context may allocate USM host memory (if corresponding extension is available), which could be more efficient than regular host memory.</summary>
    /// <param name="context">A pointer to the ov_remote_context_t instance.</param>
    /// <param name="type">Defines the element type of the tensor.</param>
    /// <param name="shape">Defines the shape of the tensor.</param>
    /// <param name="tensor">Pointer to ov_tensor_t that contains host tensor.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_remote_context_create_host_tensor(ov_remote_context* context, ov_element_type_e type, ov_shape_t shape, ov_tensor** tensor);
    
    /// <summary>Release the memory allocated by ov_remote_context_t.</summary>
    /// <param name="context">A pointer to the ov_remote_context_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_remote_context_free(ov_remote_context* context);
    
    /// <summary>Returns a string contains device-specific parameters required for low-level operations with underlying object. Parameters include device/context/surface/buffer handles, access flags, etc. Content of the returned map depends on remote execution context that is currently set on the device (working scenario). One example: "MEM_HANDLE:0x559ff6904b00;OCL_CONTEXT:0x559ff71d62f0;SHARED_MEM_TYPE:OCL_BUFFER;"</summary>
    /// <param name="tensor">Pointer to ov_tensor_t that contains host tensor.</param>
    /// <param name="size">The size of param pairs.</param>
    /// <param name="params">Param name:value list.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_remote_tensor_get_params(ov_tensor* tensor, ulong* size, byte** @params);
    
    /// <summary>Returns name of a device on which underlying object is allocated.</summary>
    /// <param name="remote_tensor">A pointer to the remote tensor instance.</param>
    /// <param name="device_name">Device name will be return.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_remote_tensor_get_device_name(ov_tensor* remote_tensor, byte** device_name);
    
    /// <summary>Get the input size of ov_compiled_model_t.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="input_size">the compiled_model's input size.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_inputs_size(ov_compiled_model* compiled_model, ulong* size);
    
    /// <summary>Get the single const input port of ov_compiled_model_t, which only support single input model.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="input_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_input(ov_compiled_model* compiled_model, ov_output_const_port** input_port);
    
    /// <summary>Get a const input port of ov_compiled_model_t by port index.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="index">input index.</param>
    /// <param name="input_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_input_by_index(ov_compiled_model* compiled_model, ulong index, ov_output_const_port** input_port);
    
    /// <summary>Get a const input port of ov_compiled_model_t by name.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="name">input tensor name (char *).</param>
    /// <param name="input_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_input_by_name(ov_compiled_model* compiled_model, byte* name, ov_output_const_port** input_port);
    
    /// <summary>Get the output size of ov_compiled_model_t.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="size">the compiled_model's output size.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_outputs_size(ov_compiled_model* compiled_model, ulong* size);
    
    /// <summary>Get the single const output port of ov_compiled_model_t, which only support single output model.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_output(ov_compiled_model* compiled_model, ov_output_const_port** output_port);
    
    /// <summary>Get a const output port of ov_compiled_model_t by port index.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="index">input index.</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_output_by_index(ov_compiled_model* compiled_model, ulong index, ov_output_const_port** output_port);
    
    /// <summary>Get a const output port of ov_compiled_model_t by name.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="name">input tensor name (char *).</param>
    /// <param name="output_port">A pointer to the ov_output_const_port_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_output_by_name(ov_compiled_model* compiled_model, byte* name, ov_output_const_port** output_port);
    
    /// <summary>Gets runtime model information from a device.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="model">A pointer to the ov_model_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_get_runtime_model(ov_compiled_model* compiled_model, ov_model** model);
    
    /// <summary>Creates an inference request object used to infer the compiled model.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="infer_request">A pointer to the ov_infer_request_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_create_infer_request(ov_compiled_model* compiled_model, ov_infer_request** infer_request);
    
    /// <summary>Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="...">variadic paramaters The format is  <char  *property_key, char* property_value>.  Supported property key please see ov_property.h.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_set_property(ov_compiled_model* compiled_model);
    
    /// <summary>Gets properties for current compiled model.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="property_key">Property key.</param>
    /// <param name="property_value">A pointer to property value.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_get_property(ov_compiled_model* compiled_model, byte* property_key, byte** property_value);
    
    /// <summary>Exports the current compiled model to an output stream `std::ostream`. The exported model can also be imported via the ov::Core::import_model method.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="export_model_path">Path to the file.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_export_model(ov_compiled_model* compiled_model, byte* export_model_path);
    
    /// <summary>Release the memory allocated by ov_compiled_model_t.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_compiled_model_free(ov_compiled_model* compiled_model);
    
    /// <summary>Returns pointer to device-specific shared context on a remote accelerator device that was used to create this CompiledModel.</summary>
    /// <param name="compiled_model">A pointer to the ov_compiled_model_t.</param>
    /// <param name="context">Return context.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_compiled_model_get_context(ov_compiled_model* compiled_model, ov_remote_context** context);
    
    /// <summary>Get version of OpenVINO.</summary>
    /// <param name="ov_version_t">a pointer to the version</param>
    [DllImport(Dll)] public static extern ov_status_e ov_get_openvino_version(ov_version* version);
    
    /// <summary>Release the memory allocated by ov_version_t.</summary>
    /// <param name="version">A pointer to the ov_version_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_version_free(ov_version* version);
    
    /// <summary>Constructs OpenVINO Core instance by default. See RegisterPlugins for more details.</summary>
    /// <param name="core">A pointer to the newly created ov_core_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_create(ov_core** core);
    
    /// <summary>Constructs OpenVINO Core instance using XML configuration file with devices description. See RegisterPlugins for more details.</summary>
    /// <param name="xml_config_file">A path to .xml file with devices to load from. If XML configuration file is not specified,  then default plugin.xml file will be used.</param>
    /// <param name="core">A pointer to the newly created ov_core_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_create_with_config(byte* xml_config_file, ov_core** core);
    
    /// <summary>Constructs OpenVINO Core instance. See RegisterPlugins for more details.</summary>
    /// <param name="xml_config_file_ws">A path to model file with unicode.</param>
    /// <param name="core">A pointer to the newly created ov_core_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_create_with_config_unicode(ushort* xml_config_file_ws, ov_core** core);
    
    /// <summary>Release the memory allocated by ov_core_t.</summary>
    /// <param name="core">A pointer to the ov_core_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_core_free(ov_core* core);
    
    /// <summary>Reads models from IR / ONNX / PDPD / TF / TFLite formats.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="model_path">Path to a model.</param>
    /// <param name="bin_path">Path to a data file.  For IR format (*.bin):   * if `bin_path` is empty, will try to read a bin file with the same name as xml and   * if the bin file with the same name is not found, will load IR without weights.  For the following file formats the `bin_path` parameter is not used:   * ONNX format (*.onnx)   * PDPD (*.pdmodel)   * TF (*.pb)   * TFLite (*.tflite)</param>
    /// <param name="model">A pointer to the newly created model.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_read_model(ov_core* core, byte* model_path, byte* bin_path, ov_model** model);
    
    /// <summary>Reads models from IR / ONNX / PDPD / TF / TFLite formats, path is unicode.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="model_path">Path to a model.</param>
    /// <param name="bin_path">Path to a data file.  For IR format (*.bin):   * if `bin_path` is empty, will try to read a bin file with the same name as xml and   * if the bin file with the same name is not found, will load IR without weights.  For the following file formats the `bin_path` parameter is not used:   * ONNX format (*.onnx)   * PDPD (*.pdmodel)   * TF (*.pb)   * TFLite (*.tflite)</param>
    /// <param name="model">A pointer to the newly created model.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_read_model_unicode(ov_core* core, ushort* model_path, ushort* bin_path, ov_model** model);
    
    /// <summary>Reads models from IR / ONNX / PDPD / TF / TFLite formats.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="model_str">String with a model in IR / ONNX / PDPD / TF / TFLite format.</param>
    /// <param name="weights">Shared pointer to a constant tensor with weights.</param>
    /// <param name="model">A pointer to the newly created model.  Reading ONNX / PDPD / TF / TFLite models does not support loading weights from the   tensors.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_read_model_from_memory(ov_core* core, byte* model_str, ov_tensor* weights, ov_model** model);
    
    /// <summary>Creates a compiled model from a source model object. Users can create as many compiled models as they need and use them simultaneously (up to the limitation of the hardware resources).</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="model">Model object acquired from Core::read_model.</param>
    /// <param name="device_name">Name of a device to load a model to.</param>
    /// <param name="property_args_size">How many properties args will be passed, each property contains 2 args: key and value.</param>
    /// <param name="compiled_model">A pointer to the newly created compiled_model.</param>
    /// <param name="property">paramater: Optional pack of pairs:  <char * property_key, char* property_value> relevant only  for this load operation operation. Supported property key please see ov_property.h.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_compile_model(ov_core* core, ov_model* model, byte* device_name, ulong property_args_size, ov_compiled_model** compiled_model);
    
    /// <summary>Reads a model and creates a compiled model from the IR/ONNX/PDPD file. This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow, especially for cases when caching is enabled and a cached model is available.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="model_path">Path to a model.</param>
    /// <param name="device_name">Name of a device to load a model to.</param>
    /// <param name="property_args_size">How many properties args will be passed, each property contains 2 args: key and value.</param>
    /// <param name="compiled_model">A pointer to the newly created compiled_model.</param>
    /// <param name="...">Optional pack of pairs:  <char * property_key, char* property_value> relevant only  for this load operation operation. Supported property key please see ov_property.h.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_compile_model_from_file(ov_core* core, byte* model_path, byte* device_name, ulong property_args_size, ov_compiled_model** compiled_model);
    
    /// <summary>Reads a model and creates a compiled model from the IR/ONNX/PDPD file. This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow, especially for cases when caching is enabled and a cached model is available.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="model_path">Path to a model.</param>
    /// <param name="device_name">Name of a device to load a model to.</param>
    /// <param name="property_args_size">How many properties args will be passed, each property contains 2 args: key and value.</param>
    /// <param name="compiled_model">A pointer to the newly created compiled_model.</param>
    /// <param name="...">Optional pack of pairs:  <char * property_key, char* property_value> relevant only  for this load operation operation. Supported property key please see ov_property.h.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_compile_model_from_file_unicode(ov_core* core, ushort* model_path, byte* device_name, ulong property_args_size, ov_compiled_model** compiled_model);
    
    /// <summary>Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="device_name">Name of a device.</param>
    /// <param name="...">variadic paramaters The format is  <char * property_key, char* property_value>.  Supported property key please see ov_property.h.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_set_property(ov_core* core, byte* device_name);
    
    /// <summary>Gets properties related to device behaviour. The method extracts information that can be set via the set_property method.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="device_name">Name of a device to get a property value.</param>
    /// <param name="property_key">Property key.</param>
    /// <param name="property_value">A pointer to property value with string format.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_get_property(ov_core* core, byte* device_name, byte* property_key, byte** property_value);
    
    /// <summary>Returns devices available for inference.</summary>
    /// <param name="core">A pointer to the ie_core_t instance.</param>
    /// <param name="devices">A pointer to the ov_available_devices_t instance.  Core objects go over all registered plugins and ask about available devices.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_get_available_devices(ov_core* core, ov_available_devices_t* devices);
    
    /// <summary>Releases memory occpuied by ov_available_devices_t</summary>
    /// <param name="devices">A pointer to the ov_available_devices_t instance.</param>
    [DllImport(Dll)] public static extern void ov_available_devices_free(ov_available_devices_t* devices);
    
    /// <summary>Imports a compiled model from the previously exported one.</summary>
    /// <param name="core">A pointer to the ov_core_t instance.</param>
    /// <param name="content">A pointer to content of the exported model.</param>
    /// <param name="content_size">Number of bytes in the exported network.</param>
    /// <param name="device_name">Name of a device to import a compiled model for.</param>
    /// <param name="compiled_model">A pointer to the newly created compiled_model.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_import_model(ov_core* core, byte* content, ulong content_size, byte* device_name, ov_compiled_model** compiled_model);
    
    /// <summary>Returns device plugins version information. Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`; in this case, std::map contains multiple entries, each per device.</summary>
    /// <param name="core">A pointer to the ov_core_t instance.</param>
    /// <param name="device_name">Device name to identify a plugin.</param>
    /// <param name="versions">A pointer to versions corresponding to device_name.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_get_versions_by_device_name(ov_core* core, byte* device_name, ov_core_version_list_t* versions);
    
    /// <summary>Releases memory occupied by ov_core_version_list_t.</summary>
    /// <param name="versions">A pointer to the ie_core_versions to free memory.</param>
    [DllImport(Dll)] public static extern void ov_core_versions_free(ov_core_version_list_t* versions);
    
    /// <summary>Creates a new remote shared context object on the specified accelerator device using specified plugin-specific low-level device API parameters (device handle, pointer, context, etc.).</summary>
    /// <param name="core">A pointer to the ov_core_t instance.</param>
    /// <param name="device_name">Device name to identify a plugin.</param>
    /// <param name="context_args_size">How many property args will be for this remote context creation.</param>
    /// <param name="context">A pointer to the newly created remote context.</param>
    /// <param name="variadic">parmameters Actual context property parameter for remote context</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_create_context(ov_core* core, byte* device_name, ulong context_args_size, ov_remote_context** context);
    
    /// <summary>Creates a compiled model from a source model within a specified remote context.</summary>
    /// <param name="core">A pointer to the ov_core_t instance.</param>
    /// <param name="model">Model object acquired from ov_core_read_model.</param>
    /// <param name="context">A pointer to the newly created remote context.</param>
    /// <param name="property_args_size">How many args will be for this compiled model.</param>
    /// <param name="compiled_model">A pointer to the newly created compiled_model.</param>
    /// <param name="variadic">parmameters Actual property parameter for remote context</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_compile_model_with_context(ov_core* core, ov_model* model, ov_remote_context* context, ulong property_args_size, ov_compiled_model** compiled_model);
    
    /// <summary>Gets a pointer to default (plugin-supplied) shared context object for the specified accelerator device.</summary>
    /// <param name="core">A pointer to the ov_core_t instance.</param>
    /// <param name="device_name">Name of a device to get a default shared context from.</param>
    /// <param name="context">A pointer to the referenced remote context.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_core_get_default_context(ov_core* core, byte* device_name, ov_remote_context** context);
    
    /// <summary>Create a ov_preprocess_prepostprocessor_t instance.</summary>
    /// <param name="model">A pointer to the ov_model_t.</param>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_create(ov_model* model, ov_preprocess_prepostprocessor** preprocess);
    
    /// <summary>Release the memory allocated by ov_preprocess_prepostprocessor_t.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_prepostprocessor_free(ov_preprocess_prepostprocessor* preprocess);
    
    /// <summary>Get the input info of ov_preprocess_prepostprocessor_t instance.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="preprocess_input_info">A pointer to the ov_preprocess_input_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_get_input_info(ov_preprocess_prepostprocessor* preprocess, ov_preprocess_input_info** preprocess_input_info);
    
    /// <summary>Get the input info of ov_preprocess_prepostprocessor_t instance by tensor name.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="tensor_name">The name of input.</param>
    /// <param name="preprocess_input_info">A pointer to the ov_preprocess_input_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_get_input_info_by_name(ov_preprocess_prepostprocessor* preprocess, byte* tensor_name, ov_preprocess_input_info** preprocess_input_info);
    
    /// <summary>Get the input info of ov_preprocess_prepostprocessor_t instance by tensor order.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="tensor_index">The order of input.</param>
    /// <param name="preprocess_input_info">A pointer to the ov_preprocess_input_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_get_input_info_by_index(ov_preprocess_prepostprocessor* preprocess, ulong tensor_index, ov_preprocess_input_info** preprocess_input_info);
    
    /// <summary>Release the memory allocated by ov_preprocess_input_info_t.</summary>
    /// <param name="preprocess_input_info">A pointer to the ov_preprocess_input_info_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_input_info_free(ov_preprocess_input_info* preprocess_input_info);
    
    /// <summary>Get a ov_preprocess_input_tensor_info_t.</summary>
    /// <param name="preprocess_input_info">A pointer to the ov_preprocess_input_info_t.</param>
    /// <param name="preprocess_input_tensor_info">A pointer to ov_preprocess_input_tensor_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_info_get_tensor_info(ov_preprocess_input_info* preprocess_input_info, ov_preprocess_input_tensor_info** preprocess_input_tensor_info);
    
    /// <summary>Release the memory allocated by ov_preprocess_input_tensor_info_t.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_input_tensor_info_free(ov_preprocess_input_tensor_info* preprocess_input_tensor_info);
    
    /// <summary>Get a ov_preprocess_preprocess_steps_t.</summary>
    /// <param name="ov_preprocess_input_info_t">A pointer to the ov_preprocess_input_info_t.</param>
    /// <param name="preprocess_input_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_info_get_preprocess_steps(ov_preprocess_input_info* preprocess_input_info, ov_preprocess_preprocess_steps** preprocess_input_steps);
    
    /// <summary>Release the memory allocated by ov_preprocess_preprocess_steps_t.</summary>
    /// <param name="preprocess_input_steps">A pointer to the ov_preprocess_preprocess_steps_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_preprocess_steps_free(ov_preprocess_preprocess_steps* preprocess_input_process_steps);
    
    /// <summary>Add resize operation to model's dimensions.</summary>
    /// <param name="preprocess_input_process_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    /// <param name="resize_algorithm">A ov_preprocess_resizeAlgorithm instance</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_resize(ov_preprocess_preprocess_steps* preprocess_input_process_steps, ov_preprocess_resize_algorithm_e resize_algorithm);
    
    /// <summary>Add scale preprocess operation. Divide each element of input by specified value.</summary>
    /// <param name="preprocess_input_process_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    /// <param name="value">Scaling value</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_scale(ov_preprocess_preprocess_steps* preprocess_input_process_steps, float value);
    
    /// <summary>Add mean preprocess operation. Subtract specified value from each element of input.</summary>
    /// <param name="preprocess_input_process_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    /// <param name="value">Value to subtract from each element.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_mean(ov_preprocess_preprocess_steps* preprocess_input_process_steps, float value);
    
    /// <summary>Crop input tensor between begin and end coordinates.</summary>
    /// <param name="preprocess_input_process_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    /// <param name="begin">Pointer to begin indexes for input tensor cropping.  Negative values represent counting elements from the end of input tensor</param>
    /// <param name="begin_size">The size of begin array</param>
    /// <param name="end">Pointer to end indexes for input tensor cropping.  End indexes are exclusive, which means values including end edge are not included in the output slice.  Negative values represent counting elements from the end of input tensor</param>
    /// <param name="end_size">The size of end array</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_crop(ov_preprocess_preprocess_steps* preprocess_input_process_steps, int* begin, int begin_size, int* end, int end_size);
    
    /// <summary>Add 'convert layout' operation to specified layout.</summary>
    /// <param name="preprocess_input_process_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    /// <param name="layout">A point to ov_layout_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_convert_layout(ov_preprocess_preprocess_steps* preprocess_input_process_steps, ov_layout* layout);
    
    /// <summary>Reverse channels operation.</summary>
    /// <param name="preprocess_input_process_steps">A pointer to ov_preprocess_preprocess_steps_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_reverse_channels(ov_preprocess_preprocess_steps* preprocess_input_process_steps);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t precesion.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="element_type">A point to element_type</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_element_type(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, ov_element_type_e element_type);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t color format.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="colorFormat">The enumerate of colorFormat</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_color_format(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, ov_color_format_e colorFormat);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t color format with subname.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="colorFormat">The enumerate of colorFormat</param>
    /// <param name="sub_names_size">The size of sub_names</param>
    /// <param name="variadic">params sub_names Optional list of sub-names assigned for each plane (e.g. "Y", "UV").</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_color_format_with_subname(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, ov_color_format_e colorFormat, ulong sub_names_size);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t spatial_static_shape.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="input_height">The height of input</param>
    /// <param name="input_width">The width of input</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_spatial_static_shape(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, ulong input_height, ulong input_width);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t memory type.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="mem_type">Memory type. Refer to ov_remote_context.h to get memory type string info.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_memory_type(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, byte* mem_type);
    
    /// <summary>Convert ov_preprocess_preprocess_steps_t element type.</summary>
    /// <param name="preprocess_input_steps">A pointer to the ov_preprocess_preprocess_steps_t.</param>
    /// <param name="element_type">preprocess input element type.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_convert_element_type(ov_preprocess_preprocess_steps* preprocess_input_process_steps, ov_element_type_e element_type);
    
    /// <summary>Convert ov_preprocess_preprocess_steps_t color.</summary>
    /// <param name="preprocess_input_steps">A pointer to the ov_preprocess_preprocess_steps_t.</param>
    /// <param name="colorFormat">The enumerate of colorFormat.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_preprocess_steps_convert_color(ov_preprocess_preprocess_steps* preprocess_input_process_steps, ov_color_format_e colorFormat);
    
    /// <summary>Helper function to reuse element type and shape from user's created tensor.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="tensor">A point to ov_tensor_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_from(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, ov_tensor* tensor);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t layout.</summary>
    /// <param name="preprocess_input_tensor_info">A pointer to the ov_preprocess_input_tensor_info_t.</param>
    /// <param name="layout">A point to ov_layout_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info* preprocess_input_tensor_info, ov_layout* layout);
    
    /// <summary>Get the output info of ov_preprocess_output_info_t instance.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="preprocess_output_info">A pointer to the ov_preprocess_output_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_get_output_info(ov_preprocess_prepostprocessor* preprocess, ov_preprocess_output_info** preprocess_output_info);
    
    /// <summary>Get the output info of ov_preprocess_output_info_t instance.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="tensor_index">The tensor index</param>
    /// <param name="preprocess_output_info">A pointer to the ov_preprocess_output_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_get_output_info_by_index(ov_preprocess_prepostprocessor* preprocess, ulong tensor_index, ov_preprocess_output_info** preprocess_output_info);
    
    /// <summary>Get the output info of ov_preprocess_output_info_t instance.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="tensor_name">The name of input.</param>
    /// <param name="preprocess_output_info">A pointer to the ov_preprocess_output_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_get_output_info_by_name(ov_preprocess_prepostprocessor* preprocess, byte* tensor_name, ov_preprocess_output_info** preprocess_output_info);
    
    /// <summary>Release the memory allocated by ov_preprocess_output_info_t.</summary>
    /// <param name="preprocess_output_info">A pointer to the ov_preprocess_output_info_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_output_info_free(ov_preprocess_output_info* preprocess_output_info);
    
    /// <summary>Get a ov_preprocess_input_tensor_info_t.</summary>
    /// <param name="preprocess_output_info">A pointer to the ov_preprocess_output_info_t.</param>
    /// <param name="preprocess_output_tensor_info">A pointer to the ov_preprocess_output_tensor_info_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_output_info_get_tensor_info(ov_preprocess_output_info* preprocess_output_info, ov_preprocess_output_tensor_info** preprocess_output_tensor_info);
    
    /// <summary>Release the memory allocated by ov_preprocess_output_tensor_info_t.</summary>
    /// <param name="preprocess_output_tensor_info">A pointer to the ov_preprocess_output_tensor_info_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_output_tensor_info_free(ov_preprocess_output_tensor_info* preprocess_output_tensor_info);
    
    /// <summary>Set ov_preprocess_input_tensor_info_t precesion.</summary>
    /// <param name="preprocess_output_tensor_info">A pointer to the ov_preprocess_output_tensor_info_t.</param>
    /// <param name="element_type">A point to element_type</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_output_set_element_type(ov_preprocess_output_tensor_info* preprocess_output_tensor_info, ov_element_type_e element_type);
    
    /// <summary>Get current input model information.</summary>
    /// <param name="preprocess_input_info">A pointer to the ov_preprocess_input_info_t.</param>
    /// <param name="preprocess_input_model_info">A pointer to the ov_preprocess_input_model_info_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_info_get_model_info(ov_preprocess_input_info* preprocess_input_info, ov_preprocess_input_model_info** preprocess_input_model_info);
    
    /// <summary>Release the memory allocated by ov_preprocess_input_model_info_t.</summary>
    /// <param name="preprocess_input_model_info">A pointer to the ov_preprocess_input_model_info_t to free memory.</param>
    [DllImport(Dll)] public static extern void ov_preprocess_input_model_info_free(ov_preprocess_input_model_info* preprocess_input_model_info);
    
    /// <summary>Set layout for model's input tensor.</summary>
    /// <param name="preprocess_input_model_info">A pointer to the ov_preprocess_input_model_info_t</param>
    /// <param name="layout">A point to ov_layout_t</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_input_model_info_set_layout(ov_preprocess_input_model_info* preprocess_input_model_info, ov_layout* layout);
    
    /// <summary>Adds pre/post-processing operations to function passed in constructor.</summary>
    /// <param name="preprocess">A pointer to the ov_preprocess_prepostprocessor_t.</param>
    /// <param name="model">A pointer to the ov_model_t.</param>
    [DllImport(Dll)] public static extern ov_status_e ov_preprocess_prepostprocessor_build(ov_preprocess_prepostprocessor* preprocess, ov_model** model);
}
