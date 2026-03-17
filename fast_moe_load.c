/*
 * fast_moe_load.c — Pre-stacked expert weight loading for MoE inference
 *
 * Eliminates Python mx.stack() overhead (540 calls/token for 60 layers × 9 components)
 * by pre-allocating STACKED buffers [K, *shape] per (layer, component).
 *
 * Each expert slot is a contiguous slice within the stacked buffer.
 * pread() fills the right slice directly — no stacking, no Python dict assembly.
 *
 * The Python side gets back pre-built expert_tensors dicts ready for compute_moe_direct.
 *
 * API:
 *   init(num_workers=8)
 *   prealloc_stacked(num_layers, K, components, packed_dir, expert_size)
 *   load_and_assemble(routing_list)  -- THE hot path
 *   get_stacked_buffers()
 *   stats()
 *   shutdown()
 *
 * Data flow:
 *   1. prealloc_stacked: creates mx.zeros([K, *shape]) per (layer, component)
 *      Stores raw pointers: slot i starts at base + i * slot_stride
 *   2. load_and_assemble: takes [(layer, [expert_indices]), ...]
 *      a. Builds pread work items (parallel, GIL released)
 *      b. Returns list of 60 dicts {comp_name: mx.array[K, ...]}
 *      The returned arrays ARE the pre-allocated stacked buffers — zero copy.
 *
 * BF16 handling:
 *   Scales/biases are stored as uint16 on disk, need .view(mx.bfloat16) in Python.
 *   We create PAIRED arrays: one uint16 for pread, one bfloat16 view of same memory.
 *   load_and_assemble returns the bfloat16 view for scales/biases components.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdatomic.h>

/* macOS QoS */
#include <sys/qos.h>
#include <pthread/qos.h>

/* ---- Constants ---- */

#define FML_MAX_LAYERS       64
#define FML_MAX_K            16      /* max active experts per token */
#define FML_MAX_COMPONENTS   9       /* gate/up/down × weight/scales/biases */
#define FML_MAX_WORKERS      32
#define FML_MAX_WORK_ITEMS   (FML_MAX_LAYERS * FML_MAX_K * FML_MAX_COMPONENTS)
#define FML_COMP_NAME_LEN    48
#define FML_PAGE_SIZE        16384   /* macOS ARM64 page size */

/* ---- Component spec ---- */

typedef struct {
    char name[FML_COMP_NAME_LEN];   /* e.g. "gate_proj.weight" */
    size_t offset;                   /* byte offset within expert block */
    size_t size;                     /* byte size per expert for this component */
    int ndim;
    int shape[4];                    /* per-expert shape (without K dimension) */
    char mx_dtype_str[16];           /* "uint32", "uint16", etc. */
    int needs_bf16_view;             /* 1 if stored as uint16 but needs bfloat16 view */
} ComponentSpec;

/* ---- Stacked buffer entry: one [K, *shape] array per (layer, component) ---- */

typedef struct {
    PyObject *mx_array;              /* The stacked mx.array [K, *shape] (pread target) */
    PyObject *mx_view_array;         /* bfloat16 view if needs_bf16_view, else same as mx_array */
    void     *data_ptr;              /* Raw CPU pointer to base of Metal buffer */
    size_t    slot_stride;           /* bytes between consecutive experts in the stacked dim */
    size_t    total_bytes;           /* K * slot_stride */
} StackedBuf;

/* ---- Per-layer file descriptor ---- */

typedef struct {
    int fd;
} LayerFd;

/* ---- Single pread work item ---- */

typedef struct {
    int    fd;
    void  *dest;
    size_t nbytes;
    off_t  file_offset;
    int    error;
    ssize_t bytes_read;
} WorkItem;

/* ---- Worker thread context ---- */

typedef struct {
    pthread_t       thread;
    int             worker_id;
    int             running;

    WorkItem       *items;
    int             item_count;

    pthread_mutex_t work_mutex;
    pthread_cond_t  work_cond;
    int             has_work;

    pthread_mutex_t *done_mutex;
    pthread_cond_t  *done_cond;
    atomic_int      *completed_count;
} WorkerCtx;

/* ---- Module state ---- */

typedef struct {
    /* Worker pool */
    WorkerCtx      *workers;
    int             num_workers;

    /* Stacked buffers: [layer][component] — each is [K, *shape] */
    StackedBuf      bufs[FML_MAX_LAYERS][FML_MAX_COMPONENTS];

    /* Component specs */
    ComponentSpec   comp_specs[FML_MAX_COMPONENTS];
    int             num_comps;
    size_t          expert_size;     /* total bytes per expert block in packed file */

    /* Layer file descriptors */
    LayerFd         layer_fds[FML_MAX_LAYERS];
    int             num_layers;
    int             K;               /* experts per token (stacking dimension) */

    /* Pre-built Python dicts per layer: list of dicts */
    PyObject       *py_layer_dicts;  /* Python list of 60 dicts */

    /* Completion sync */
    pthread_mutex_t done_mutex;
    pthread_cond_t  done_cond;
    atomic_int      completed_count;

    int             initialized;
    int             preallocated;

    /* Stats */
    long long       total_loads;
    long long       total_bytes;
    long long       total_calls;
} ModuleState;

static ModuleState g_state = {0};

/* ---- Worker thread function ---- */

static void *worker_func(void *arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;

    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);

    while (ctx->running) {
        pthread_mutex_lock(&ctx->work_mutex);
        while (!ctx->has_work && ctx->running) {
            pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
        }
        if (!ctx->running) {
            pthread_mutex_unlock(&ctx->work_mutex);
            break;
        }

        WorkItem *items = ctx->items;
        int count = ctx->item_count;
        ctx->has_work = 0;
        pthread_mutex_unlock(&ctx->work_mutex);

        /* Execute pread calls — no GIL, no Python involvement */
        for (int i = 0; i < count; i++) {
            WorkItem *wi = &items[i];
            ssize_t nread = pread(wi->fd, wi->dest, wi->nbytes, wi->file_offset);
            if (nread < 0) {
                wi->error = errno;
                wi->bytes_read = -1;
            } else if ((size_t)nread < wi->nbytes) {
                size_t total = (size_t)nread;
                while (total < wi->nbytes) {
                    ssize_t n = pread(wi->fd,
                                      (char *)wi->dest + total,
                                      wi->nbytes - total,
                                      wi->file_offset + (off_t)total);
                    if (n <= 0) {
                        wi->error = (n < 0) ? errno : EIO;
                        wi->bytes_read = (ssize_t)total;
                        break;
                    }
                    total += (size_t)n;
                }
                if (total == wi->nbytes) {
                    wi->error = 0;
                    wi->bytes_read = (ssize_t)total;
                }
            } else {
                wi->error = 0;
                wi->bytes_read = nread;
            }
        }

        /* Signal completion */
        atomic_fetch_add(ctx->completed_count, count);
        pthread_mutex_lock(ctx->done_mutex);
        pthread_cond_signal(ctx->done_cond);
        pthread_mutex_unlock(ctx->done_mutex);
    }

    return NULL;
}

/* ---- init(num_workers=8) ---- */

static PyObject *fml_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"num_workers", NULL};
    int num_workers = 8;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &num_workers))
        return NULL;

    if (g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "fast_moe_load already initialized");
        return NULL;
    }

    if (num_workers < 1 || num_workers > FML_MAX_WORKERS) {
        PyErr_Format(PyExc_ValueError, "num_workers must be 1-%d", FML_MAX_WORKERS);
        return NULL;
    }

    pthread_mutex_init(&g_state.done_mutex, NULL);
    pthread_cond_init(&g_state.done_cond, NULL);
    atomic_store(&g_state.completed_count, 0);

    g_state.num_workers = num_workers;
    g_state.workers = (WorkerCtx *)calloc(num_workers, sizeof(WorkerCtx));
    if (!g_state.workers) {
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < num_workers; i++) {
        WorkerCtx *w = &g_state.workers[i];
        w->worker_id = i;
        w->running = 1;
        w->has_work = 0;
        w->items = NULL;
        w->item_count = 0;
        w->done_mutex = &g_state.done_mutex;
        w->done_cond = &g_state.done_cond;
        w->completed_count = &g_state.completed_count;

        pthread_mutex_init(&w->work_mutex, NULL);
        pthread_cond_init(&w->work_cond, NULL);

        int rc = pthread_create(&w->thread, NULL, worker_func, w);
        if (rc != 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to create worker thread %d: %s", i, strerror(rc));
            for (int j = 0; j < i; j++) {
                g_state.workers[j].running = 0;
                pthread_mutex_lock(&g_state.workers[j].work_mutex);
                g_state.workers[j].has_work = 1;
                pthread_cond_signal(&g_state.workers[j].work_cond);
                pthread_mutex_unlock(&g_state.workers[j].work_mutex);
                pthread_join(g_state.workers[j].thread, NULL);
                pthread_mutex_destroy(&g_state.workers[j].work_mutex);
                pthread_cond_destroy(&g_state.workers[j].work_cond);
            }
            free(g_state.workers);
            g_state.workers = NULL;
            return NULL;
        }
    }

    g_state.initialized = 1;
    g_state.total_loads = 0;
    g_state.total_bytes = 0;
    g_state.total_calls = 0;

    Py_RETURN_NONE;
}

/* ---- Helper: get data pointer from an mx.array via buffer protocol ---- */

static void *get_mx_data_ptr(PyObject *mx_array) {
    Py_buffer view;
    void *ptr = NULL;

    /* Try writable first */
    if (PyObject_GetBuffer(mx_array, &view, PyBUF_WRITABLE | PyBUF_SIMPLE) == 0) {
        ptr = view.buf;
        PyBuffer_Release(&view);
        return ptr;
    }
    PyErr_Clear();

    /* Try read-only */
    if (PyObject_GetBuffer(mx_array, &view, PyBUF_SIMPLE) == 0) {
        ptr = view.buf;
        PyBuffer_Release(&view);
        return ptr;
    }
    PyErr_Clear();

    /* Try memoryview */
    PyObject *py_memview = PyMemoryView_FromObject(mx_array);
    if (py_memview) {
        Py_buffer *mv_buf = PyMemoryView_GET_BUFFER(py_memview);
        ptr = mv_buf->buf;
        Py_DECREF(py_memview);
        return ptr;
    }
    PyErr_Clear();

    /* Last resort: numpy */
    PyObject *np_mod = PyImport_ImportModule("numpy");
    if (np_mod) {
        PyObject *np_asarray = PyObject_GetAttrString(np_mod, "asarray");
        if (np_asarray) {
            PyObject *np_arr = PyObject_CallFunctionObjArgs(np_asarray, mx_array, NULL);
            if (np_arr) {
                if (PyObject_GetBuffer(np_arr, &view, PyBUF_WRITABLE | PyBUF_SIMPLE) == 0) {
                    ptr = view.buf;
                    PyBuffer_Release(&view);
                }
                if (!ptr) {
                    PyErr_Clear();
                    if (PyObject_GetBuffer(np_arr, &view, PyBUF_SIMPLE) == 0) {
                        ptr = view.buf;
                        PyBuffer_Release(&view);
                    }
                }
                Py_DECREF(np_arr);
            }
            Py_DECREF(np_asarray);
        }
        Py_DECREF(np_mod);
    }

    if (!ptr) {
        PyErr_Clear();
        PyErr_SetString(PyExc_RuntimeError,
                        "Cannot get data pointer from mx.array via buffer protocol");
    }

    return ptr;
}

/*
 * ---- prealloc_stacked(num_layers, K, components, packed_dir, expert_size) ----
 *
 * Pre-allocate STACKED mx.array buffers: [K, *shape] per (layer, component).
 * K expert slots are contiguous in memory — no mx.stack needed at inference time.
 *
 * For BF16 components (scales/biases stored as uint16):
 *   Creates paired arrays: uint16 [K, *shape] for pread + bfloat16 view of same memory.
 *
 * components: list of dicts, each with:
 *   - name: str (e.g. "gate_proj.weight")
 *   - offset: int (byte offset within expert block)
 *   - size: int (byte size per expert)
 *   - shape: list of int (per-expert shape, without K dim)
 *   - dtype: str (e.g. "uint32", "uint16")
 *   - needs_bf16_view: bool (optional, default False)
 *
 * Returns: list of num_layers dicts, each {comp_name: mx.array[K, ...]}
 *          where scales/biases are already bfloat16 views.
 */

static PyObject *fml_prealloc_stacked(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"num_layers", "K", "components",
                             "packed_dir", "expert_size", NULL};
    int num_layers = 0;
    int K = 0;
    PyObject *py_components = NULL;
    const char *packed_dir = NULL;
    size_t expert_size = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOs|n", kwlist,
                                     &num_layers, &K,
                                     &py_components, &packed_dir, &expert_size))
        return NULL;

    if (!g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() first");
        return NULL;
    }

    if (g_state.preallocated) {
        PyErr_SetString(PyExc_RuntimeError, "Already pre-allocated. Call shutdown() first.");
        return NULL;
    }

    if (num_layers <= 0 || num_layers > FML_MAX_LAYERS) {
        PyErr_Format(PyExc_ValueError, "num_layers must be 1-%d, got %d",
                     FML_MAX_LAYERS, num_layers);
        return NULL;
    }
    if (K <= 0 || K > FML_MAX_K) {
        PyErr_Format(PyExc_ValueError, "K must be 1-%d, got %d", FML_MAX_K, K);
        return NULL;
    }

    if (!PyList_Check(py_components)) {
        PyErr_SetString(PyExc_TypeError, "components must be a list");
        return NULL;
    }

    Py_ssize_t num_comps = PyList_Size(py_components);
    if (num_comps <= 0 || num_comps > FML_MAX_COMPONENTS) {
        PyErr_Format(PyExc_ValueError, "components count must be 1-%d", FML_MAX_COMPONENTS);
        return NULL;
    }

    /* Parse component specs */
    g_state.num_comps = (int)num_comps;
    for (Py_ssize_t ci = 0; ci < num_comps; ci++) {
        PyObject *comp = PyList_GetItem(py_components, ci);
        if (!comp || !PyDict_Check(comp)) {
            PyErr_Format(PyExc_TypeError, "components[%zd] must be a dict", ci);
            return NULL;
        }

        ComponentSpec *cs = &g_state.comp_specs[ci];

        /* name */
        PyObject *py_name = PyDict_GetItemString(comp, "name");
        if (!py_name) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'name'", ci);
            return NULL;
        }
        const char *name = PyUnicode_AsUTF8(py_name);
        if (!name) return NULL;
        strncpy(cs->name, name, FML_COMP_NAME_LEN - 1);
        cs->name[FML_COMP_NAME_LEN - 1] = '\0';

        /* offset */
        PyObject *py_off = PyDict_GetItemString(comp, "offset");
        if (!py_off) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'offset'", ci);
            return NULL;
        }
        cs->offset = (size_t)PyLong_AsUnsignedLongLong(py_off);

        /* size */
        PyObject *py_sz = PyDict_GetItemString(comp, "size");
        if (!py_sz) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'size'", ci);
            return NULL;
        }
        cs->size = (size_t)PyLong_AsUnsignedLongLong(py_sz);

        /* shape */
        PyObject *py_shape = PyDict_GetItemString(comp, "shape");
        if (!py_shape || !PyList_Check(py_shape)) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing or invalid 'shape'", ci);
            return NULL;
        }
        cs->ndim = (int)PyList_Size(py_shape);
        if (cs->ndim <= 0 || cs->ndim > 4) {
            PyErr_Format(PyExc_ValueError, "components[%zd] shape dims must be 1-4", ci);
            return NULL;
        }
        for (int d = 0; d < cs->ndim; d++) {
            cs->shape[d] = (int)PyLong_AsLong(PyList_GetItem(py_shape, d));
        }

        /* dtype */
        PyObject *py_dtype = PyDict_GetItemString(comp, "dtype");
        if (!py_dtype) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'dtype'", ci);
            return NULL;
        }
        const char *dtype_str = PyUnicode_AsUTF8(py_dtype);
        if (!dtype_str) return NULL;
        strncpy(cs->mx_dtype_str, dtype_str, 15);
        cs->mx_dtype_str[15] = '\0';

        /* needs_bf16_view */
        PyObject *py_bf16 = PyDict_GetItemString(comp, "needs_bf16_view");
        cs->needs_bf16_view = (py_bf16 && PyObject_IsTrue(py_bf16)) ? 1 : 0;
    }

    g_state.expert_size = expert_size;
    g_state.num_layers = num_layers;
    g_state.K = K;

    /* Open packed layer files */
    for (int li = 0; li < num_layers; li++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, li);

        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            PyErr_Format(PyExc_OSError, "Cannot open %s: %s", path, strerror(errno));
            for (int k = 0; k < li; k++) {
                close(g_state.layer_fds[k].fd);
                g_state.layer_fds[k].fd = -1;
            }
            return NULL;
        }
        g_state.layer_fds[li].fd = fd;
    }

    /* Import mlx */
    PyObject *mx_module = PyImport_ImportModule("mlx.core");
    if (!mx_module) return NULL;

    PyObject *mx_zeros = PyObject_GetAttrString(mx_module, "zeros");
    PyObject *mx_eval = PyObject_GetAttrString(mx_module, "eval");
    if (!mx_zeros || !mx_eval) {
        Py_XDECREF(mx_zeros);
        Py_XDECREF(mx_eval);
        Py_DECREF(mx_module);
        PyErr_SetString(PyExc_RuntimeError, "Cannot find mx.zeros or mx.eval");
        return NULL;
    }

    /* Build the list of per-layer dicts */
    PyObject *layer_list = PyList_New(num_layers);
    if (!layer_list) {
        Py_DECREF(mx_zeros);
        Py_DECREF(mx_eval);
        Py_DECREF(mx_module);
        return NULL;
    }

    int alloc_ok = 1;

    for (int li = 0; li < num_layers && alloc_ok; li++) {
        PyObject *layer_dict = PyDict_New();
        if (!layer_dict) { alloc_ok = 0; break; }

        for (int ci = 0; ci < (int)num_comps && alloc_ok; ci++) {
            ComponentSpec *cs = &g_state.comp_specs[ci];

            /*
             * Build stacked shape: (K, shape[0], shape[1], ...)
             * e.g. weight [1024, 512] -> [K, 1024, 512]
             */
            int stacked_ndim = 1 + cs->ndim;
            PyObject *py_shape = PyTuple_New(stacked_ndim);
            PyTuple_SET_ITEM(py_shape, 0, PyLong_FromLong(K));
            for (int d = 0; d < cs->ndim; d++) {
                PyTuple_SET_ITEM(py_shape, 1 + d, PyLong_FromLong(cs->shape[d]));
            }

            /* Get mx dtype object */
            PyObject *mx_dtype = PyObject_GetAttrString(mx_module, cs->mx_dtype_str);
            if (!mx_dtype) {
                PyErr_Format(PyExc_ValueError, "Unknown mlx dtype: mx.%s", cs->mx_dtype_str);
                Py_DECREF(py_shape);
                alloc_ok = 0;
                Py_DECREF(layer_dict);
                break;
            }

            /* Create mx.zeros(stacked_shape, dtype=dtype) */
            PyObject *kwargs_dict = PyDict_New();
            PyDict_SetItemString(kwargs_dict, "dtype", mx_dtype);
            PyObject *call_args = PyTuple_Pack(1, py_shape);
            PyObject *arr = PyObject_Call(mx_zeros, call_args, kwargs_dict);

            Py_DECREF(call_args);
            Py_DECREF(kwargs_dict);
            Py_DECREF(mx_dtype);
            Py_DECREF(py_shape);

            if (!arr) {
                alloc_ok = 0;
                Py_DECREF(layer_dict);
                break;
            }

            /* mx.eval(arr) to force Metal buffer allocation */
            PyObject *eval_args = PyTuple_Pack(1, arr);
            PyObject *eval_result = PyObject_Call(mx_eval, eval_args, NULL);
            Py_DECREF(eval_args);
            Py_XDECREF(eval_result);
            if (!eval_result) {
                Py_DECREF(arr);
                alloc_ok = 0;
                Py_DECREF(layer_dict);
                break;
            }

            /* Get raw data pointer */
            void *ptr = get_mx_data_ptr(arr);
            if (!ptr) {
                PyErr_Format(PyExc_RuntimeError,
                             "Cannot get data pointer for %s (layer=%d)",
                             cs->name, li);
                Py_DECREF(arr);
                alloc_ok = 0;
                Py_DECREF(layer_dict);
                break;
            }

            /* Create bfloat16 view if needed */
            PyObject *view_arr = NULL;
            if (cs->needs_bf16_view) {
                /* arr.view(mx.bfloat16) — reinterpret uint16 bits as bfloat16 */
                PyObject *bf16_dtype = PyObject_GetAttrString(mx_module, "bfloat16");
                if (!bf16_dtype) {
                    Py_DECREF(arr);
                    alloc_ok = 0;
                    Py_DECREF(layer_dict);
                    break;
                }
                PyObject *view_method = PyObject_GetAttrString(arr, "view");
                if (!view_method) {
                    Py_DECREF(bf16_dtype);
                    Py_DECREF(arr);
                    alloc_ok = 0;
                    Py_DECREF(layer_dict);
                    break;
                }
                view_arr = PyObject_CallFunctionObjArgs(view_method, bf16_dtype, NULL);
                Py_DECREF(view_method);
                Py_DECREF(bf16_dtype);
                if (!view_arr) {
                    Py_DECREF(arr);
                    alloc_ok = 0;
                    Py_DECREF(layer_dict);
                    break;
                }
            } else {
                view_arr = arr;
                Py_INCREF(view_arr);
            }

            /* Store in C-level table */
            StackedBuf *sb = &g_state.bufs[li][ci];
            sb->mx_array = arr;          /* owns reference (uint16 or native) */
            sb->mx_view_array = view_arr; /* owns reference (bfloat16 view or same) */
            sb->data_ptr = ptr;
            sb->slot_stride = cs->size;   /* bytes per expert for this component */
            sb->total_bytes = (size_t)K * cs->size;

            /* Store view_arr in layer dict (this is what Python sees) */
            PyObject *py_comp_key = PyUnicode_FromString(cs->name);
            PyDict_SetItem(layer_dict, py_comp_key, view_arr);
            Py_DECREF(py_comp_key);
        }

        if (alloc_ok) {
            /* PyList_SET_ITEM steals the reference */
            PyList_SET_ITEM(layer_list, li, layer_dict);
        } else {
            /* layer_dict already decref'd in error path above */
        }
    }

    Py_DECREF(mx_zeros);
    Py_DECREF(mx_eval);
    Py_DECREF(mx_module);

    if (!alloc_ok) {
        /* Cleanup any partially allocated arrays */
        for (int li = 0; li < num_layers; li++) {
            for (int ci = 0; ci < (int)num_comps; ci++) {
                StackedBuf *sb = &g_state.bufs[li][ci];
                Py_XDECREF(sb->mx_array);
                Py_XDECREF(sb->mx_view_array);
                sb->mx_array = NULL;
                sb->mx_view_array = NULL;
                sb->data_ptr = NULL;
            }
        }
        for (int li = 0; li < num_layers; li++) {
            if (g_state.layer_fds[li].fd >= 0) {
                close(g_state.layer_fds[li].fd);
                g_state.layer_fds[li].fd = -1;
            }
        }
        Py_DECREF(layer_list);
        return NULL;
    }

    g_state.py_layer_dicts = layer_list;  /* We own this reference */
    g_state.preallocated = 1;

    /* Return the list (caller gets a new reference) */
    Py_INCREF(layer_list);
    return layer_list;
}

/*
 * ---- load_and_assemble(routing_list) ----
 *
 * THE HOT PATH. Single C call replaces:
 *   - Python loop over 60 layers
 *   - 540 mx.stack() calls
 *   - 540 dict insertions
 *   - 540 .view(mx.bfloat16) calls
 *
 * Input: routing_list — list of (layer_idx, expert_indices_list) tuples
 *   e.g. [(0, [23, 45, 120, 7]), (1, [12, 67, 200, 3]), ...]
 *
 * For each (layer, experts):
 *   1. pread K experts into stacked buffer slots (parallel pthreads, GIL released)
 *   2. The pre-allocated stacked arrays [K, *shape] are already filled in-place
 *
 * Returns: the pre-built layer_dicts list (same objects as from prealloc_stacked).
 *          The arrays within are now filled with the correct expert data.
 *
 * The caller uses: expert_tensors = result[layer_idx]
 *                  y = compute_moe_direct(h_post, remapped_inds, expert_tensors, ...)
 *
 * IMPORTANT: The returned dicts always contain the same mx.array objects.
 * Each call to load_and_assemble overwrites the buffer contents in-place.
 * The caller must consume the data before the next call.
 */

static PyObject *fml_load_and_assemble(PyObject *self, PyObject *args) {
    PyObject *py_routing_list = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_routing_list))
        return NULL;

    if (!g_state.initialized || !g_state.preallocated) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() and prealloc_stacked() first");
        return NULL;
    }

    if (!PyList_Check(py_routing_list) && !PyTuple_Check(py_routing_list)) {
        PyErr_SetString(PyExc_TypeError, "routing_list must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_entries = PySequence_Size(py_routing_list);
    if (num_entries == 0) {
        /* Return the pre-built dicts unchanged */
        Py_INCREF(g_state.py_layer_dicts);
        return g_state.py_layer_dicts;
    }

    /*
     * Build work items: for each (layer, [experts]), each expert, each component.
     * Work items = num_entries * K * num_comps (worst case).
     * We pread directly into the stacked buffer at slot_stride offsets.
     */
    int max_items = (int)(num_entries * g_state.K * g_state.num_comps);
    if (max_items > FML_MAX_WORK_ITEMS) {
        PyErr_Format(PyExc_OverflowError,
                     "Too many work items: %d (max %d)", max_items, FML_MAX_WORK_ITEMS);
        return NULL;
    }

    WorkItem *items = (WorkItem *)calloc(max_items, sizeof(WorkItem));
    if (!items) {
        PyErr_NoMemory();
        return NULL;
    }

    int item_idx = 0;

    for (Py_ssize_t ei = 0; ei < num_entries; ei++) {
        PyObject *entry = PySequence_GetItem(py_routing_list, ei);
        if (!entry) { free(items); return NULL; }

        /* Unpack (layer_idx, expert_indices_list) */
        PyObject *py_layer_idx = NULL;
        PyObject *py_expert_list = NULL;

        if (PyTuple_Check(entry) && PyTuple_Size(entry) == 2) {
            py_layer_idx = PyTuple_GetItem(entry, 0);
            py_expert_list = PyTuple_GetItem(entry, 1);
        } else if (PyList_Check(entry) && PyList_Size(entry) == 2) {
            py_layer_idx = PyList_GetItem(entry, 0);
            py_expert_list = PyList_GetItem(entry, 1);
        } else {
            Py_DECREF(entry);
            free(items);
            PyErr_SetString(PyExc_TypeError,
                            "Each routing entry must be (layer_idx, [expert_indices])");
            return NULL;
        }

        int layer_idx = (int)PyLong_AsLong(py_layer_idx);
        if (layer_idx < 0 || layer_idx >= g_state.num_layers) {
            Py_DECREF(entry);
            free(items);
            PyErr_Format(PyExc_ValueError, "layer_idx %d out of range (0-%d)",
                         layer_idx, g_state.num_layers - 1);
            return NULL;
        }

        if (!PyList_Check(py_expert_list) && !PyTuple_Check(py_expert_list)) {
            Py_DECREF(entry);
            free(items);
            PyErr_SetString(PyExc_TypeError, "expert_indices must be a list or tuple");
            return NULL;
        }

        Py_ssize_t num_experts = PySequence_Size(py_expert_list);
        if (num_experts > g_state.K) {
            Py_DECREF(entry);
            free(items);
            PyErr_Format(PyExc_ValueError,
                         "Too many experts %zd for layer %d (K=%d)",
                         num_experts, layer_idx, g_state.K);
            return NULL;
        }

        int fd = g_state.layer_fds[layer_idx].fd;

        for (Py_ssize_t si = 0; si < num_experts; si++) {
            PyObject *py_eidx = PySequence_GetItem(py_expert_list, si);
            int expert_idx = (int)PyLong_AsLong(py_eidx);
            Py_DECREF(py_eidx);

            off_t expert_base = (off_t)expert_idx * (off_t)g_state.expert_size;

            for (int ci = 0; ci < g_state.num_comps; ci++) {
                ComponentSpec *cs = &g_state.comp_specs[ci];
                StackedBuf *sb = &g_state.bufs[layer_idx][ci];

                WorkItem *wi = &items[item_idx++];
                wi->fd = fd;
                /* Destination: slot si within the stacked buffer */
                wi->dest = (char *)sb->data_ptr + (size_t)si * sb->slot_stride;
                wi->nbytes = cs->size;
                wi->file_offset = expert_base + (off_t)cs->offset;
                wi->error = 0;
                wi->bytes_read = 0;
            }
        }

        Py_DECREF(entry);
    }

    if (item_idx == 0) {
        free(items);
        Py_INCREF(g_state.py_layer_dicts);
        return g_state.py_layer_dicts;
    }

    /* Distribute work items to workers (round-robin) */
    WorkItem **worker_items = (WorkItem **)calloc(g_state.num_workers, sizeof(WorkItem *));
    int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
    if (!worker_items || !worker_counts) {
        free(items);
        free(worker_items);
        free(worker_counts);
        PyErr_NoMemory();
        return NULL;
    }

    int *worker_alloc = (int *)calloc(g_state.num_workers, sizeof(int));
    for (int i = 0; i < item_idx; i++) {
        worker_alloc[i % g_state.num_workers]++;
    }
    for (int w = 0; w < g_state.num_workers; w++) {
        if (worker_alloc[w] > 0) {
            worker_items[w] = (WorkItem *)calloc(worker_alloc[w], sizeof(WorkItem));
            if (!worker_items[w]) {
                for (int k = 0; k < w; k++) free(worker_items[k]);
                free(worker_items);
                free(worker_counts);
                free(worker_alloc);
                free(items);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }
    free(worker_alloc);

    for (int i = 0; i < item_idx; i++) {
        int wid = i % g_state.num_workers;
        worker_items[wid][worker_counts[wid]++] = items[i];
    }

    /* Reset completion counter */
    atomic_store(&g_state.completed_count, 0);

    /* Dispatch to workers */
    for (int w = 0; w < g_state.num_workers; w++) {
        WorkerCtx *wk = &g_state.workers[w];
        pthread_mutex_lock(&wk->work_mutex);
        wk->items = worker_items[w];
        wk->item_count = worker_counts[w];
        wk->has_work = (worker_counts[w] > 0) ? 1 : 0;
        if (wk->has_work)
            pthread_cond_signal(&wk->work_cond);
        pthread_mutex_unlock(&wk->work_mutex);
    }

    /* Release GIL and wait for all workers to complete */
    int total_expected = item_idx;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_state.done_mutex);
    while (atomic_load(&g_state.completed_count) < total_expected) {
        pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
    }
    pthread_mutex_unlock(&g_state.done_mutex);
    Py_END_ALLOW_THREADS

    /* Check for errors */
    int errors = 0;
    long long bytes_loaded = 0;
    for (int w = 0; w < g_state.num_workers; w++) {
        for (int j = 0; j < worker_counts[w]; j++) {
            if (worker_items[w][j].error != 0) {
                errors++;
            } else {
                bytes_loaded += worker_items[w][j].bytes_read;
            }
        }
    }

    /* Cleanup per-worker arrays */
    for (int w = 0; w < g_state.num_workers; w++) {
        free(worker_items[w]);
    }
    free(worker_items);
    free(worker_counts);
    free(items);

    /* Update stats */
    g_state.total_loads += num_entries;
    g_state.total_bytes += bytes_loaded;
    g_state.total_calls++;

    if (errors > 0) {
        PyErr_Format(PyExc_IOError,
                     "pread failed for %d/%d work items", errors, item_idx);
        return NULL;
    }

    /* Return the pre-built layer dicts.
     * The stacked buffers have been filled in-place with expert data.
     * The dicts contain the same mx.array objects as always — their
     * contents have been updated via pread into the underlying Metal buffer. */
    Py_INCREF(g_state.py_layer_dicts);
    return g_state.py_layer_dicts;
}

/*
 * ---- load_experts_direct(load_list) ----
 *
 * Compatibility function: same interface as fast_weight_load.load_experts
 * but writes into the stacked buffers.
 *
 * load_list: list of (layer_idx, expert_idx, slot_idx) tuples
 * Fills slot slot_idx within the stacked buffers for layer_idx.
 */

static PyObject *fml_load_experts_direct(PyObject *self, PyObject *args) {
    PyObject *py_load_list = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_load_list))
        return NULL;

    if (!g_state.initialized || !g_state.preallocated) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() and prealloc_stacked() first");
        return NULL;
    }

    if (!PyList_Check(py_load_list) && !PyTuple_Check(py_load_list)) {
        PyErr_SetString(PyExc_TypeError, "load_list must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_entries = PySequence_Size(py_load_list);
    if (num_entries == 0) {
        return PyLong_FromLong(0);
    }

    int total_items = (int)(num_entries * g_state.num_comps);
    if (total_items > FML_MAX_WORK_ITEMS) {
        PyErr_Format(PyExc_OverflowError,
                     "Too many work items: %d (max %d)", total_items, FML_MAX_WORK_ITEMS);
        return NULL;
    }

    WorkItem *items = (WorkItem *)calloc(total_items, sizeof(WorkItem));
    if (!items) {
        PyErr_NoMemory();
        return NULL;
    }

    int item_idx = 0;
    for (Py_ssize_t i = 0; i < num_entries; i++) {
        PyObject *entry = PySequence_GetItem(py_load_list, i);
        if (!entry) { free(items); return NULL; }

        int layer_idx, expert_idx, slot_idx;
        if (!PyArg_ParseTuple(entry, "iii", &layer_idx, &expert_idx, &slot_idx)) {
            Py_DECREF(entry);
            free(items);
            return NULL;
        }
        Py_DECREF(entry);

        if (layer_idx < 0 || layer_idx >= g_state.num_layers) {
            PyErr_Format(PyExc_ValueError, "layer_idx %d out of range", layer_idx);
            free(items);
            return NULL;
        }
        if (slot_idx < 0 || slot_idx >= g_state.K) {
            PyErr_Format(PyExc_ValueError, "slot_idx %d out of range (0-%d)",
                         slot_idx, g_state.K - 1);
            free(items);
            return NULL;
        }

        int fd = g_state.layer_fds[layer_idx].fd;
        off_t expert_base = (off_t)expert_idx * (off_t)g_state.expert_size;

        for (int ci = 0; ci < g_state.num_comps; ci++) {
            ComponentSpec *cs = &g_state.comp_specs[ci];
            StackedBuf *sb = &g_state.bufs[layer_idx][ci];

            WorkItem *wi = &items[item_idx++];
            wi->fd = fd;
            wi->dest = (char *)sb->data_ptr + (size_t)slot_idx * sb->slot_stride;
            wi->nbytes = cs->size;
            wi->file_offset = expert_base + (off_t)cs->offset;
            wi->error = 0;
            wi->bytes_read = 0;
        }
    }

    /* Distribute to workers */
    WorkItem **worker_items = (WorkItem **)calloc(g_state.num_workers, sizeof(WorkItem *));
    int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
    if (!worker_items || !worker_counts) {
        free(items);
        free(worker_items);
        free(worker_counts);
        PyErr_NoMemory();
        return NULL;
    }

    int *worker_alloc = (int *)calloc(g_state.num_workers, sizeof(int));
    for (int i = 0; i < item_idx; i++) {
        worker_alloc[i % g_state.num_workers]++;
    }
    for (int w = 0; w < g_state.num_workers; w++) {
        if (worker_alloc[w] > 0) {
            worker_items[w] = (WorkItem *)calloc(worker_alloc[w], sizeof(WorkItem));
            if (!worker_items[w]) {
                for (int k = 0; k < w; k++) free(worker_items[k]);
                free(worker_items);
                free(worker_counts);
                free(worker_alloc);
                free(items);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }
    free(worker_alloc);

    for (int i = 0; i < item_idx; i++) {
        int wid = i % g_state.num_workers;
        worker_items[wid][worker_counts[wid]++] = items[i];
    }

    atomic_store(&g_state.completed_count, 0);

    for (int w = 0; w < g_state.num_workers; w++) {
        WorkerCtx *wk = &g_state.workers[w];
        pthread_mutex_lock(&wk->work_mutex);
        wk->items = worker_items[w];
        wk->item_count = worker_counts[w];
        wk->has_work = (worker_counts[w] > 0) ? 1 : 0;
        if (wk->has_work)
            pthread_cond_signal(&wk->work_cond);
        pthread_mutex_unlock(&wk->work_mutex);
    }

    int total_expected = item_idx;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_state.done_mutex);
    while (atomic_load(&g_state.completed_count) < total_expected) {
        pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
    }
    pthread_mutex_unlock(&g_state.done_mutex);
    Py_END_ALLOW_THREADS

    int errors = 0;
    long long bytes_loaded = 0;
    for (int w = 0; w < g_state.num_workers; w++) {
        for (int j = 0; j < worker_counts[w]; j++) {
            if (worker_items[w][j].error != 0) {
                errors++;
            } else {
                bytes_loaded += worker_items[w][j].bytes_read;
            }
        }
    }

    for (int w = 0; w < g_state.num_workers; w++) {
        free(worker_items[w]);
    }
    free(worker_items);
    free(worker_counts);
    free(items);

    g_state.total_loads += num_entries;
    g_state.total_bytes += bytes_loaded;

    if (errors > 0) {
        PyErr_Format(PyExc_IOError, "pread failed for %d/%d work items", errors, item_idx);
        return NULL;
    }

    return PyLong_FromLong(item_idx);
}

/* ---- get_stacked_buffers() ---- */

static PyObject *fml_get_stacked_buffers(PyObject *self, PyObject *args) {
    if (!g_state.preallocated || !g_state.py_layer_dicts) {
        PyErr_SetString(PyExc_RuntimeError, "No buffers pre-allocated");
        return NULL;
    }
    Py_INCREF(g_state.py_layer_dicts);
    return g_state.py_layer_dicts;
}

/* ---- stats() ---- */

static PyObject *fml_stats(PyObject *self, PyObject *args) {
    return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:n, s:L, s:L, s:L}",
                         "initialized", g_state.initialized,
                         "preallocated", g_state.preallocated,
                         "num_workers", g_state.num_workers,
                         "num_layers", g_state.num_layers,
                         "K", g_state.K,
                         "expert_size", (Py_ssize_t)g_state.expert_size,
                         "total_loads", g_state.total_loads,
                         "total_bytes", g_state.total_bytes,
                         "total_calls", g_state.total_calls);
}

/* ---- shutdown() ---- */

static PyObject *fml_shutdown(PyObject *self, PyObject *args) {
    if (!g_state.initialized) {
        Py_RETURN_NONE;
    }

    /* Stop worker threads */
    if (g_state.workers) {
        for (int i = 0; i < g_state.num_workers; i++) {
            WorkerCtx *w = &g_state.workers[i];
            pthread_mutex_lock(&w->work_mutex);
            w->running = 0;
            w->has_work = 1;
            pthread_cond_signal(&w->work_cond);
            pthread_mutex_unlock(&w->work_mutex);
        }
        for (int i = 0; i < g_state.num_workers; i++) {
            pthread_join(g_state.workers[i].thread, NULL);
            pthread_mutex_destroy(&g_state.workers[i].work_mutex);
            pthread_cond_destroy(&g_state.workers[i].work_cond);
        }
        free(g_state.workers);
        g_state.workers = NULL;
    }

    /* Release stacked buffer references */
    if (g_state.preallocated) {
        for (int li = 0; li < g_state.num_layers; li++) {
            for (int ci = 0; ci < g_state.num_comps; ci++) {
                StackedBuf *sb = &g_state.bufs[li][ci];
                Py_XDECREF(sb->mx_array);
                Py_XDECREF(sb->mx_view_array);
                sb->mx_array = NULL;
                sb->mx_view_array = NULL;
                sb->data_ptr = NULL;
            }
        }
        Py_XDECREF(g_state.py_layer_dicts);
        g_state.py_layer_dicts = NULL;
        g_state.preallocated = 0;
    }

    /* Close layer file descriptors */
    for (int i = 0; i < g_state.num_layers; i++) {
        if (g_state.layer_fds[i].fd >= 0) {
            close(g_state.layer_fds[i].fd);
            g_state.layer_fds[i].fd = -1;
        }
    }
    g_state.num_layers = 0;

    pthread_mutex_destroy(&g_state.done_mutex);
    pthread_cond_destroy(&g_state.done_cond);

    g_state.initialized = 0;
    g_state.num_workers = 0;
    g_state.num_comps = 0;
    g_state.expert_size = 0;
    g_state.K = 0;
    g_state.total_loads = 0;
    g_state.total_bytes = 0;
    g_state.total_calls = 0;

    Py_RETURN_NONE;
}

/* ---- Module definition ---- */

static PyMethodDef fml_methods[] = {
    {"init", (PyCFunction)fml_init, METH_VARARGS | METH_KEYWORDS,
     "init(num_workers=8) -- Create persistent worker thread pool"},
    {"prealloc_stacked", (PyCFunction)fml_prealloc_stacked, METH_VARARGS | METH_KEYWORDS,
     "prealloc_stacked(num_layers, K, components, packed_dir, expert_size)\n"
     "Pre-allocate STACKED mx.array Metal buffers [K, *shape] per (layer, comp).\n"
     "BF16 components get paired uint16 + bfloat16 view arrays.\n"
     "Returns: list of num_layers dicts {comp_name: mx.array[K, ...]}"},
    {"load_and_assemble", fml_load_and_assemble, METH_VARARGS,
     "load_and_assemble(routing_list)\n"
     "THE HOT PATH. Fills stacked Metal buffers via parallel pread, GIL released.\n"
     "routing_list: [(layer_idx, [expert_indices]), ...]\n"
     "Returns: list of dicts (pre-allocated, filled in-place)"},
    {"load_experts_direct", fml_load_experts_direct, METH_VARARGS,
     "load_experts_direct(load_list)\n"
     "Compatibility: fill stacked buffers slot-by-slot.\n"
     "load_list: [(layer_idx, expert_idx, slot_idx), ...]\n"
     "Returns: number of read operations completed"},
    {"get_stacked_buffers", fml_get_stacked_buffers, METH_NOARGS,
     "get_stacked_buffers() -- Return the pre-allocated stacked buffer dicts"},
    {"stats", fml_stats, METH_NOARGS,
     "stats() -- Return diagnostic counters"},
    {"shutdown", fml_shutdown, METH_NOARGS,
     "shutdown() -- Stop workers, release buffers, close files"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fml_module = {
    PyModuleDef_HEAD_INIT,
    "fast_moe_load",
    "Pre-stacked expert weight loading for MoE inference.\n"
    "\n"
    "Eliminates Python mx.stack() overhead by pre-allocating [K, *shape]\n"
    "Metal buffers. pread fills each expert slot in-place.\n"
    "Returns ready-to-use expert_tensors dicts for compute_moe_direct.\n"
    "\n"
    "Key advantage over fast_weight_load: no Python loop to assemble\n"
    "expert_tensors dicts. One C call does I/O for all 60 layers.",
    -1,
    fml_methods
};

PyMODINIT_FUNC PyInit_fast_moe_load(void) {
    PyObject *m = PyModule_Create(&fml_module);
    if (!m) return NULL;

    PyModule_AddIntConstant(m, "MAX_LAYERS", FML_MAX_LAYERS);
    PyModule_AddIntConstant(m, "MAX_K", FML_MAX_K);
    PyModule_AddIntConstant(m, "MAX_COMPONENTS", FML_MAX_COMPONENTS);
    PyModule_AddIntConstant(m, "MAX_WORKERS", FML_MAX_WORKERS);
    PyModule_AddIntConstant(m, "PAGE_SIZE", FML_PAGE_SIZE);

    return m;
}
