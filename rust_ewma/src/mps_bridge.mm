#import <Metal/Metal.h>
#include <Python.h>
#include <torch/extension.h>

extern "C" {
void *get_mtl_buffer_and_offset(void *py_obj_ptr, uint64_t *offset) {
  PyObject *obj = (PyObject *)py_obj_ptr;
  if (!obj)
    return nullptr;

  // Check is_mps
  // t.is_mps
  PyObject *is_mps = PyObject_GetAttrString(obj, "is_mps");
  if (!is_mps) {
    PyErr_Clear();
    return nullptr;
  }
  if (!PyObject_IsTrue(is_mps)) {
    Py_DECREF(is_mps);
    return nullptr;
  }
  Py_DECREF(is_mps);

  // Get storage()
  PyObject *storage = PyObject_CallMethod(obj, "untyped_storage", NULL);
  if (!storage) {
    PyErr_Clear();
    return nullptr;
  }

  // Get data_ptr()
  PyObject *ptr_obj = PyObject_CallMethod(storage, "data_ptr", NULL);
  void *ptr = nullptr;
  if (ptr_obj) {
    ptr = PyLong_AsVoidPtr(ptr_obj);
    Py_DECREF(ptr_obj);
  }
  Py_DECREF(storage);

  if (!ptr) {
    return nullptr;
  }

  // Get storage_offset()
  PyObject *s_offset_obj = PyObject_CallMethod(obj, "storage_offset", NULL);
  long long s_offset = 0;
  if (s_offset_obj) {
    s_offset = PyLong_AsLongLong(s_offset_obj);
    Py_DECREF(s_offset_obj);
  } else {
    PyErr_Clear();
  }

  // itemsize (element_size)
  // Try element_size() method first
  long long itemsize = 4;
  PyObject *es = PyObject_CallMethod(obj, "element_size", NULL);
  if (es) {
    itemsize = PyLong_AsLongLong(es);
    Py_DECREF(es);
  } else {
    PyErr_Clear();
    // Try itemsize attr (numpy compat)
    PyObject *itemsize_obj = PyObject_GetAttrString(obj, "itemsize");
    if (itemsize_obj) {
      itemsize = PyLong_AsLongLong(itemsize_obj);
      Py_DECREF(itemsize_obj);
    } else {
      PyErr_Clear();
    }
  }

  *offset = (uint64_t)(s_offset * itemsize);

  return (__bridge void *)(__bridge id<MTLBuffer>)ptr;
}

// Type definition for the Rust callback
typedef void (*RustEncoderCallback)(void *cmd_buffer, void *context);

void dispatch_on_mps(RustEncoderCallback callback, void *context) {
  dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
  dispatch_sync(serialQueue, ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
      callback((__bridge void *)commandBuffer, context);
      // DO NOT commit here
      // torch::mps::commit();
    }
  });
}
}
