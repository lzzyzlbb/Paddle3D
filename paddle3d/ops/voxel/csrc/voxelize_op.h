#pragma once

#include <vector>

#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
#else
#include <algorithm>
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
#endif
