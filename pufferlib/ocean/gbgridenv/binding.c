#include <Python.h>
#include <limits.h>
#include "gbgridenv.h"

#define Env GbGridEnv
static PyObject* vec_get_grid(PyObject* self, PyObject* args);
static PyObject* vec_put_grid(PyObject* self, PyObject* args, PyObject* kwargs);

#define MY_GET
#define MY_PUT
#define MY_METHODS {"vec_get_grid", (PyCFunction)vec_get_grid, METH_VARARGS, "Get grid"}, \
    {"vec_put_grid", (PyCFunction)vec_put_grid, METH_VARARGS | METH_KEYWORDS, "Put grid"}

#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    int size = (int)unpack(kwargs, "size");
    if (size <= 0) {
        size = GBGRID_DEFAULT_SIZE;
    }
    int agent_energy = 50;
    PyObject* energy_obj = PyDict_GetItemString(kwargs, "agent_energy");
    if (energy_obj && energy_obj != Py_None) {
        if (!PyLong_Check(energy_obj)) {
            PyErr_SetString(PyExc_TypeError, "agent_energy must be an integer");
            return 1;
        }
        long energy_val = PyLong_AsLong(energy_obj);
        if (energy_val < 0) {
            energy_val = 0;
        }
        if (energy_val > INT_MAX) {
            PyErr_SetString(PyExc_ValueError, "agent_energy is out of range");
            return 1;
        }
        agent_energy = (int)energy_val;
    }
    gbgrid_alloc(env, size, size);
    env->agent_initial_energy = agent_energy;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    return 0;
}

static PyObject* my_get(PyObject* dict, Env* env) {
    if (!env || env->width <= 0 || env->height <= 0 || !env->grid) {
        PyDict_SetItemString(dict, "width", PyLong_FromLong(0));
        PyDict_SetItemString(dict, "height", PyLong_FromLong(0));
        PyDict_SetItemString(dict, "cell_types", Py_None);
        PyDict_SetItemString(dict, "soil", Py_None);
        Py_INCREF(Py_None);
        Py_INCREF(Py_None);
        return dict;
    }

    PyDict_SetItemString(dict, "width", PyLong_FromLong(env->width));
    PyDict_SetItemString(dict, "height", PyLong_FromLong(env->height));

    npy_intp dims2[2] = {env->height, env->width};
    PyObject* cell_types = PyArray_SimpleNew(2, dims2, NPY_UINT8);
    if (!cell_types) {
        return NULL;
    }
    unsigned char* cell_data = (unsigned char*)PyArray_DATA((PyArrayObject*)cell_types);

    npy_intp dims3[3] = {env->height, env->width, 5};
    PyObject* soil = PyArray_SimpleNew(3, dims3, NPY_FLOAT32);
    if (!soil) {
        Py_DECREF(cell_types);
        return NULL;
    }
    float* soil_data = (float*)PyArray_DATA((PyArrayObject*)soil);

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            GbGridCell* cell = &env->grid[idx];
            cell_data[idx] = (unsigned char)cell->type;

            int soil_idx = idx * 5;
            soil_data[soil_idx + 0] = cell->soil.water;
            soil_data[soil_idx + 1] = cell->soil.organic_material;
            soil_data[soil_idx + 2] = cell->soil.nutr_a;
            soil_data[soil_idx + 3] = cell->soil.nutr_b;
            soil_data[soil_idx + 4] = cell->soil.nutr_c;
        }
    }

    PyDict_SetItemString(dict, "cell_types", cell_types);
    PyDict_SetItemString(dict, "soil", soil);
    Py_DECREF(cell_types);
    Py_DECREF(soil);
    return dict;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    if (!kwargs) {
        return 0;
    }

    PyObject* size_obj = PyDict_GetItemString(kwargs, "size");
    if (size_obj && size_obj != Py_None) {
        if (!PyLong_Check(size_obj)) {
            PyErr_SetString(PyExc_TypeError, "size must be an integer");
            return 1;
        }
        int size = (int)PyLong_AsLong(size_obj);
        if (size > 0 && gbgrid_alloc(env, size, size) != 0) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate grid");
            return 1;
        }
    }

    PyArrayObject* cell_types = NULL;
    PyArrayObject* soil = NULL;

    PyObject* cell_types_obj = PyDict_GetItemString(kwargs, "cell_types");
    if (cell_types_obj && cell_types_obj != Py_None) {
        cell_types = (PyArrayObject*)PyArray_FROM_OTF(cell_types_obj, NPY_UINT8, NPY_ARRAY_C_CONTIGUOUS);
        if (!cell_types) {
            PyErr_SetString(PyExc_TypeError, "cell_types must be a uint8 array");
            return 1;
        }
        if (PyArray_NDIM(cell_types) != 2) {
            PyErr_SetString(PyExc_ValueError, "cell_types must be 2D");
            Py_DECREF(cell_types);
            return 1;
        }
    }

    PyObject* soil_obj = PyDict_GetItemString(kwargs, "soil");
    if (soil_obj && soil_obj != Py_None) {
        soil = (PyArrayObject*)PyArray_FROM_OTF(soil_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
        if (!soil) {
            Py_XDECREF(cell_types);
            PyErr_SetString(PyExc_TypeError, "soil must be a float32 array");
            return 1;
        }
        if (PyArray_NDIM(soil) != 3) {
            PyErr_SetString(PyExc_ValueError, "soil must be 3D (H, W, 5)");
            Py_XDECREF(cell_types);
            Py_DECREF(soil);
            return 1;
        }
    }

    int width = env->width;
    int height = env->height;
    if ((width <= 0 || height <= 0) && (cell_types || soil)) {
        if (cell_types) {
            npy_intp* dims = PyArray_DIMS(cell_types);
            height = (int)dims[0];
            width = (int)dims[1];
        } else if (soil) {
            npy_intp* dims = PyArray_DIMS(soil);
            height = (int)dims[0];
            width = (int)dims[1];
        }
        if (gbgrid_alloc(env, width, height) != 0) {
            Py_XDECREF(cell_types);
            Py_XDECREF(soil);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate grid");
            return 1;
        }
    }

    if (cell_types) {
        npy_intp* dims = PyArray_DIMS(cell_types);
        if (dims[0] != env->height || dims[1] != env->width) {
            Py_XDECREF(cell_types);
            Py_XDECREF(soil);
            PyErr_SetString(PyExc_ValueError, "cell_types shape must match grid");
            return 1;
        }
    }
    if (soil) {
        npy_intp* dims = PyArray_DIMS(soil);
        if (dims[0] != env->height || dims[1] != env->width || dims[2] != 5) {
            Py_XDECREF(cell_types);
            Py_XDECREF(soil);
            PyErr_SetString(PyExc_ValueError, "soil shape must be (H, W, 5)");
            return 1;
        }
    }

    unsigned char* cell_data = cell_types ? (unsigned char*)PyArray_DATA(cell_types) : NULL;
    float* soil_data = soil ? (float*)PyArray_DATA(soil) : NULL;

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            GbGridCell* cell = &env->grid[idx];
            if (cell_data) {
                cell->type = (GbGridCellType)cell_data[idx];
            }
            if (soil_data) {
                int soil_idx = idx * 5;
                cell->soil.water = soil_data[soil_idx + 0];
                cell->soil.organic_material = soil_data[soil_idx + 1];
                cell->soil.nutr_a = soil_data[soil_idx + 2];
                cell->soil.nutr_b = soil_data[soil_idx + 3];
                cell->soil.nutr_c = soil_data[soil_idx + 4];
            }
        }
    }

    Py_XDECREF(cell_types);
    Py_XDECREF(soil);
    return 0;
}

static PyObject* vec_get_grid(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get_grid requires 2 arguments");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = (int)PyLong_AsLong(env_id_arg);
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    PyObject* dict = PyDict_New();
    my_get(dict, vec->envs[env_id]);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return dict;
}

static PyObject* vec_put_grid(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_put_grid requires 2 positional arguments");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = (int)PyLong_AsLong(env_id_arg);
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    PyObject* empty_args = PyTuple_New(0);
    if (my_put(vec->envs[env_id], empty_args, kwargs) != 0) {
        Py_DECREF(empty_args);
        return NULL;
    }
    Py_DECREF(empty_args);
    Py_RETURN_NONE;
}
