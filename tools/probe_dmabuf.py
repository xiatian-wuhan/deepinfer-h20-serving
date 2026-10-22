"""真实测一次 GPUDirect RDMA 需要的 dma-buf 导出：cuMemCreate -> cuMemMap -> DMA_BUF_FD。
   如果这里能拿到 fd，说明 cuDeviceGetAttribute(124)==0 是在说谎。"""
import traceback
from cuda.bindings import driver as cu

def call(name, fn, *a, **kw):
    try:
        r = fn(*a, **kw)
    except Exception:
        print(f"{name}: EXCEPTION"); traceback.print_exc(limit=1); return None
    if isinstance(r, tuple):
        err = r[0]
        if int(getattr(err, "value", err)) != 0:
            print(f"{name}: err={err}"); return None
        return r[1] if len(r) == 2 else r[1:]
    return r

call("cuInit", cu.cuInit, 0)
dev = call("cuDeviceGet", cu.cuDeviceGet, 0)
print("device:", dev)
call("primaryCtxRetain", cu.cuDevicePrimaryCtxRetain, dev)
try:
    r = cu.cuDevicePrimaryCtxRetain(dev); ctx = r[1]
    cu.cuCtxSetCurrent(ctx)
    print("ctx current: OK")
except Exception:
    print("ctx: using existing"); traceback.print_exc(limit=1)

print("attr 124 (DMA_BUF_SUPPORTED) =", cu.cuDeviceGetAttribute(124, dev))

prop = cu.CUmemAllocationProp()
prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_DMA_BUF_FD
prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
prop.location.id = int(dev)
gran = call("cuMemGetAllocationGranularity", cu.cuMemGetAllocationGranularity, prop,
            cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
print("granularity:", gran)
if gran is None: raise SystemExit(1)

err, handle = (None, None)
r = cu.cuMemCreate(gran, prop, 0)
print("cuMemCreate ->", r)
handle = r[1]
ptr = cu.cuMemAddressReserve(gran, 0, 0, 0)[1]
print("reserve ->", hex(int(ptr)))
print("cuMemMap ->", cu.cuMemMap(ptr, gran, 0, handle, 0))
ad = cu.CUmemAccessDesc()
ad.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
ad.location.id = int(dev)
ad.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
print("cuMemSetAccess ->", cu.cuMemSetAccess(ptr, gran, [ad], 1))
r = cu.cuMemGetHandleForAddressRange(ptr, gran,
        cu.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)
print(">>> cuMemGetHandleForAddressRange(DMA_BUF_FD) ->", r)
