#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>

/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for rainflow counting in C.";

static char rainflow_docstring[] =
    "Rainflow cycle counting, compiled C version.\n"
    "\n"
    "**Usage:**\n"
    "\n"
    "rf = rainflow(peaks, getoffsets=False)\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "peaks : 1d array_like\n"
    "    Vector of alternating peaks (as returned by\n"
    "    :func:`pyyeti.cyclecount.findap`, for example)\n"
    "getoffsets : bool; optional\n"
    "    If True, the tuple ``(rf, os)`` is returned; otherwise, only\n"
    "    `rf` is returned.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "rf : 2d ndarray\n"
    "    n x 3 matrix with the rainflow cycle count information\n"
    "    ``[amp, mean, count]``:\n"
    "\n"
    "        - amp is the cycle amplitude (half the peak-to-peak range)\n"
    "        - mean is mean of the cycle\n"
    "        - count is either 0.5 or 1.0 depending on whether it's\n"
    "          half or full cycle\n"
    "\n"
    "os : 2d ndarray; optional\n"
    "    n x 2 matrix of cycle offsets ``[start, stop]``. Only returned\n"
    "    if `getoffsets` is True. The start and stop values are:\n"
    "\n"
    "        - start is the offset into `peaks` for start of cycle\n"
    "        - stop is the offset into `peaks` for end of cycle\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This algorithm is derived from reference [#rain1]_ and is very\n"
    "fast. The plain Python version uses the same logic.\n"
    "\n"
    "References\n"
    "----------\n"
    ".. [#rain1] \"Standard Practices for Cycle Counting in Fatigue\n"
    "       Analysis\", ASTM E 1049 - 85 (Reapproved 2005).\n"
    "\n"
    "Examples\n"
    "--------\n"
    "Run the example from the ASTM paper:\n"
    "\n"
    ">>> from pyyeti.rainflow.c_rain import rainflow\n"
    ">>> rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])\n"
    "array([[ 1.5, -0.5,  0.5],\n"
    "       [ 2. , -1. ,  0.5],\n"
    "       [ 2. ,  1. ,  1. ],\n"
    "       [ 4. ,  1. ,  0.5],\n"
    "       [ 4.5,  0.5,  0.5],\n"
    "       [ 4. ,  0. ,  0.5],\n"
    "       [ 3. ,  1. ,  0.5]])\n"
    "\n"
    "With offsets:\n"
    "\n"
    ">>> rf, os = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2],\n"
    "...                   getoffsets=True)\n"
    ">>> rf\n"
    "array([[ 1.5, -0.5,  0.5],\n"
    "       [ 2. , -1. ,  0.5],\n"
    "       [ 2. ,  1. ,  1. ],\n"
    "       [ 4. ,  1. ,  0.5],\n"
    "       [ 4.5,  0.5,  0.5],\n"
    "       [ 4. ,  0. ,  0.5],\n"
    "       [ 3. ,  1. ,  0.5]])\n"
    ">>> os              # doctest: +ELLIPSIS\n"
    "array([[0, 1],\n"
    "       [1, 2],\n"
    "       [4, 5],\n"
    "       [2, 3],\n"
    "       [3, 6],\n"
    "       [6, 7],\n"
    "       [7, 8]]...)\n";

/* Available functions */
static PyObject *rainflow(PyObject *self, PyObject *args, PyObject *keywds);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"rainflow", (PyCFunction)rainflow,
     METH_VARARGS | METH_KEYWORDS, rainflow_docstring},

    {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef rain_module = {
    PyModuleDef_HEAD_INIT,
    "c_rain",   /* name of module */
    module_docstring, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    module_methods
};

/* Initialize the module */
PyMODINIT_FUNC PyInit_c_rain(void)
{
    PyObject *m = PyModule_Create(&rain_module);
    if (m == NULL)
        return NULL;
    /* Load `numpy` functionality. */
    import_array();
    return m;
}

/* Choose a faster routine (by 60-70% on one machine) or
   a more memory efficient routine. Define the following
   macro to use the faster routine. */
#define USE_FASTER_RAINFLOW_ROUTINE

static PyObject *rainflow1(PyArrayObject *peaks_array, npy_intp L);
static PyObject *rainflow2(PyArrayObject *peaks_array, npy_intp L);

/* Reference 1: "Standard Practices for Cycle Counting in Fatigue
   Analysis", ASTM E 1049 - 85 (Reapproved 2005). */
static PyObject *rainflow(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *peaks_obj;
    int getoffsets = 0;
    static char *kwlist[] = {"peaks", "getoffsets", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|p", kwlist,
                                     &peaks_obj, &getoffsets))
      return NULL;

    /* Interpret the input object as a numpy array. */
    PyArrayObject *peaks_array=NULL;
    peaks_array = (PyArrayObject *)PyArray_FROM_OTF(peaks_obj, NPY_DOUBLE,
                                                    NPY_ARRAY_IN_ARRAY);

    if (peaks_array == NULL) return NULL;

    /* check number of dimensions and vector length */
    npy_intp L;
    int ndim = PyArray_NDIM(peaks_array);
    if (ndim == 1)
        L = PyArray_DIM(peaks_array, 0);
    else
        L = 0;
    if (L < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "`peaks` must be a real vector with length >= 2");
	Py_DECREF(peaks_array);
        return NULL;
    }

    if (getoffsets)
      return rainflow2(peaks_array, L);
    return rainflow1(peaks_array, L);
}

/* returns 3-columns matrix: [amplitude, mean, count] */
static PyObject *rainflow1(PyArrayObject *peaks_array, npy_intp L)
{
    /* prepare other pointers so the 'fail' section is safe */
    double *pts=NULL;              /* work buffer for peak values */
    PyArrayObject *rf_array=NULL;  /* return object for the rainflow count */
    PyArrayObject *srf=NULL;       /* sliced version of rf_array */

    /* Get pointer to the data as C-types. */
    double *peaks = (double*)PyArray_DATA(peaks_array);

    /* --------------------------------------------------------------- */
    pts = calloc(L, sizeof(double));
    if (pts == NULL) goto fail;

    npy_intp fullcyclesp1 = 1;  /* number of full cycles + 1 */
    npy_intp j, k;
    double X, Y;

#ifndef USE_FASTER_RAINFLOW_ROUTINE
    /* pass one:  count cycles */
    j = -1;
    for (k=0; k<L; ++k) {
      /* step 1 from [1]: */
      pts[++j] = peaks[k];
      /* step 2 from [1]: */
      while (j > 1) {
        /* step 3 from [1]: */
        Y = fabs(pts[j-2]-pts[j-1]);
        X = fabs(pts[j-1]-pts[j]);
        if (X < Y) break;
        if (j == 2) {
          /* step 5 from [1]: */
          /* [count Y as half cycle] */
          pts[0] = pts[1];  /* discard j-2 pt */
          pts[1] = pts[2];
          j = 1;
        }
        else {
          /* step 4 from [1]: */
          /* [count Y as full cycle] */
          ++fullcyclesp1;
          pts[j-2] = pts[j];  /* discard j-2, j-1 pts */
          j -= 2;
        }
      }
    }
    /* step 6 from [1]: */
    /* [count all ranges in pts as half cycles] */
#endif

    /* pass two:  store results */
    npy_intp dims[2] = {L-fullcyclesp1, 3};
    rf_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (rf_array == NULL) goto fail;
    double *rf = (double *)PyArray_DATA(rf_array);

    j = -1;
    for (k=0; k<L; ++k) {
      /* step 1 from [1]: */
      pts[++j] = peaks[k];
      /* step 2 from [1]: */
      while (j > 1) {
        /* step 3 from [1]: */
        Y = fabs(pts[j-2]-pts[j-1]);
        X = fabs(pts[j-1]-pts[j]);
        if (X < Y) break;
        if (j == 2) {
          /* step 5 from [1]: */
          /* [count Y as half cycle] */
          *rf++ = Y/2;
          *rf++ = (pts[0]+pts[1])/2;
          *rf++ = 0.5;
          pts[0] = pts[1];  /* discard j-2 pt */
          pts[1] = pts[2];
          j = 1;
        }
        else {
          /* step 4 from [1]: */
          /* [count Y as full cycle] */
#ifdef USE_FASTER_RAINFLOW_ROUTINE
          ++fullcyclesp1;
#endif
          *rf++ = Y/2;
          *rf++ = (pts[j-2]+pts[j-1])/2;
          *rf++ = 1.0;
          pts[j-2] = pts[j];  /* discard j-2, j-1 pts */
          j -= 2;
        }
      }
    }
    /* step 6 from [1]: */
    /* [count all ranges in pts as half cycles] */
    double A=pts[0], B;
    for (k=0; k<j; ++k) {
      B = pts[k+1];
      *rf++ = fabs(A-B)/2;
      *rf++ = (A+B)/2;
      *rf++ = 0.5;
      A = B;
    }

    /* Clean up. */
    Py_DECREF(peaks_array);
    free(pts);

#ifdef USE_FASTER_RAINFLOW_ROUTINE
    if (fullcyclesp1 > 1) {
      /* slice to smaller array and return */
      PyObject* stop = PyLong_FromSsize_t(L-fullcyclesp1);
      PyObject* slice = PySlice_New(NULL, stop, NULL);
      srf = (PyArrayObject *)PyObject_GetItem((PyObject *)rf_array, slice);
      Py_DECREF(stop);
      Py_DECREF(slice);
      if (srf == NULL) goto fail;
      Py_DECREF(rf_array);
      return Py_BuildValue("N", srf);
    }
#endif
    return Py_BuildValue("N", rf_array);
    /* --------------------------------------------------------------- */

fail:
    Py_XDECREF(peaks_array);
    free(pts);
    Py_XDECREF(rf_array);
    Py_XDECREF(srf);
    return NULL;
}

/* returns tuple:
   3-column matrix: [amplitude, mean, count] (double)
   2-column matrix: [start, stop] (int) */
static PyObject *rainflow2(PyArrayObject *peaks_array, npy_intp L)
{
    /* prepare other pointers so the 'fail' section is safe */
    double *pts=NULL;              /* work buffer for peak values */
    npy_intp *cycle_index=NULL;    /* work buffer for cycle offsets */
    PyArrayObject *rf_array=NULL;  /* return object for the rainflow count */
    PyArrayObject *os_array=NULL;  /* optional return object for the offsets */
    PyArrayObject *srf=NULL;       /* sliced version of rf_array */
    PyArrayObject *sos=NULL;       /* sliced version of os_array */

    /* Get pointer to the data as C-types. */
    double *peaks = (double*)PyArray_DATA(peaks_array);

    /* --------------------------------------------------------------- */
    pts = calloc(L, sizeof(double));
    cycle_index = calloc(L, sizeof(npy_intp));
    if (pts == NULL || cycle_index == NULL) goto fail;

    npy_intp fullcyclesp1 = 1;  /* number of full cycles + 1 */
    npy_intp j, k;
    double X, Y;

#ifndef USE_FASTER_RAINFLOW_ROUTINE
    /* pass one:  count cycles */
    j = -1;
    for (k=0; k<L; ++k) {
      /* step 1 from [1]: */
      pts[++j] = peaks[k];
      /* step 2 from [1]: */
      while (j > 1) {
        /* step 3 from [1]: */
        Y = fabs(pts[j-2]-pts[j-1]);
        X = fabs(pts[j-1]-pts[j]);
        if (X < Y) break;
        if (j == 2) {
          /* step 5 from [1]: */
          /* [count Y as half cycle] */
          pts[0] = pts[1];  /* discard j-2 pt */
          pts[1] = pts[2];
          j = 1;
        }
        else {
          /* step 4 from [1]: */
          /* [count Y as full cycle] */
          ++fullcyclesp1;
          pts[j-2] = pts[j];  /* discard j-2, j-1 pts */
          j -= 2;
        }
      }
    }
    /* step 6 from [1]: */
    /* [count all ranges in pts as half cycles] */
#endif

    /* pass two:  store results */
    npy_intp dims[2] = {L-fullcyclesp1, 3};
    rf_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (rf_array == NULL) goto fail;
    double *rf = (double *)PyArray_DATA(rf_array);

    dims[1] = 2;
    os_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INTP);
    if (os_array == NULL) goto fail;
    npy_intp *os = (npy_intp *)PyArray_DATA(os_array);

    j = -1;
    for (k=0; k<L; ++k) {
      /* step 1 from [1]: */
      pts[++j] = peaks[k];
      cycle_index[j] = k;
      /* step 2 from [1]: */
      while (j > 1) {
        /* step 3 from [1]: */
        Y = fabs(pts[j-2]-pts[j-1]);
        X = fabs(pts[j-1]-pts[j]);
        if (X < Y) break;
        if (j == 2) {
          /* step 5 from [1]: */
          /* [count Y as half cycle] */
          *rf++ = Y/2;
          *rf++ = (pts[0]+pts[1])/2;
          *rf++ = 0.5;
          *os++ = cycle_index[0];
          *os++ = cycle_index[1];
          pts[0] = pts[1];  /* discard j-2 pt */
          pts[1] = pts[2];
          cycle_index[0] = cycle_index[1];
          cycle_index[1] = cycle_index[2];
          j = 1;
        }
        else {
          /* step 4 from [1]: */
          /* [count Y as full cycle] */
#ifdef USE_FASTER_RAINFLOW_ROUTINE
          ++fullcyclesp1;
#endif
          *rf++ = Y/2;
          *rf++ = (pts[j-2]+pts[j-1])/2;
          *rf++ = 1.0;
          *os++ = cycle_index[j-2];
          *os++ = cycle_index[j-1];
          pts[j-2] = pts[j];  /* discard j-2, j-1 pts */
          cycle_index[j-2] = cycle_index[j];
          j -= 2;
        }
      }
    }
    /* step 6 from [1]: */
    /* [count all ranges in pts as half cycles] */
    double A=pts[0], B;
    for (k=0; k<j; ++k) {
      B = pts[k+1];
      *rf++ = fabs(A-B)/2;
      *rf++ = (A+B)/2;
      *rf++ = 0.5;
      *os++ = cycle_index[k];
      *os++ = cycle_index[k+1];
      A = B;
    }

    /* Clean up. */
    Py_DECREF(peaks_array);
    free(pts);
    free(cycle_index);

#ifdef USE_FASTER_RAINFLOW_ROUTINE
    if (fullcyclesp1 > 1) {
      /* slice to smaller array and return */
      PyObject* stop = PyLong_FromSsize_t(L-fullcyclesp1);
      PyObject* slice = PySlice_New(NULL, stop, NULL);
      srf = (PyArrayObject *)PyObject_GetItem((PyObject *)rf_array, slice);
      if (srf)
	sos = (PyArrayObject *)PyObject_GetItem((PyObject *)os_array, slice);
      Py_DECREF(stop);
      Py_DECREF(slice);
      if (srf == NULL || sos == NULL) goto fail;
      Py_DECREF(rf_array);
      Py_DECREF(os_array);
      return Py_BuildValue("NN", srf, sos);
    }
#endif
    return Py_BuildValue("NN", rf_array, os_array);
    /* --------------------------------------------------------------- */

fail:
    Py_XDECREF(peaks_array);
    free(pts);
    free(cycle_index);
    Py_XDECREF(rf_array);
    Py_XDECREF(os_array);
    Py_XDECREF(srf);
    Py_XDECREF(sos);
    return NULL;
}
