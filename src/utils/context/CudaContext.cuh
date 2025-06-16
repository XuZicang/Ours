#ifndef __CUDA_CONTEXT_CUH__
#define __CUDA_CONTEXT_CUH__

#include <cassert>
#include <cstdio>
// #if !defined(DISABLE_MGPU)
// #include <moderngpu/context.hxx>
// #endif
#include <vector>
#include "CudaContextUtils.h"
#include "../GPUTimer.cuh"
#include "../GPUUtil.cuh"
#include "../../Meta.h"
#define CUDA_CONTEXT_PROFILE
class CudaContextProfiler
{
public:
	CudaContextProfiler() { memory_operations_time_ = 0; }

	void StartTimer(cudaStream_t stream=0)
	{
#if defined(CUDA_CONTEXT_PROFILE)
		timer.StartTimer(stream);
#endif
	}
	void EndTimer(cudaStream_t stream=0)
	{
#if defined(CUDA_CONTEXT_PROFILE)
		timer.EndTimer(stream);
		memory_operations_time_ += timer.GetElapsedMilliSeconds();
#endif
	}
	double GetMemoryOperationsTime() const { return memory_operations_time_; }

private:
	double memory_operations_time_;
	GPUTimer timer;
};

// A basic implementation of CudaContext.
// If DISABLE_MGPU is enabled, we inherit the memory operation API in
// standard_context_t so that the memory allocation in library calls of
// moderngpu can use the memory allocated in this CudaContext. The memory
// operations are basic: malloc and release device memory if the APIs are called

class CudaContext
{
public:
	CudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream) : dev_mem_(dev_mem), stream_(stream) {}
	cudaStream_t Stream() { return stream_; }
	virtual void *Malloc(size_t size)
	{
		cuda_context_profiler_.StartTimer(stream_);
		void *ret = SafeMalloc(size);
		dev_mem_->Consume(size);
		cuda_context_profiler_.EndTimer(stream_);
		return ret;
	}
	virtual void Free(void *p, size_t size)
	{
		cuda_context_profiler_.StartTimer(stream_);
		SafeFree(p);
		dev_mem_->Release(size);
		cuda_context_profiler_.EndTimer(stream_);
	}
	DeviceMemoryInfo *GetDeviceMemoryInfo() const { return dev_mem_; }

	void PrintProfileResult() const
	{
#if defined(CUDA_CONTEXT_PROFILE)
		std::cout << "dev_id=" << dev_mem_->GetDevId() << ",memory_operations_time="
				  << cuda_context_profiler_.GetMemoryOperationsTime() << "ms"
				  << std::endl;
#endif
	}

protected:
	virtual void *DirectMalloc(size_t size)
	{
		void *ret = NULL;
		CUDA_ERROR(cudaMalloc(&ret, size));
		return ret;
	}

	virtual void DirectFree(void *p) { CUDA_ERROR(cudaFree(p)); }

	virtual void *SafeMalloc(size_t size)
	{
		if (dev_mem_->IsAvailable(size))
		{
			return DirectMalloc(size);
		}
		else
		{
			fprintf(stderr, "Insufficient device memory\n");
			void *ret = NULL;
			// allocate from unified memory
			CUDA_ERROR(cudaMallocManaged(&ret, size));
			CUDA_ERROR(cudaMemPrefetchAsync(ret, size, dev_mem_->GetDevId()));
			return ret;
		}
	}

	virtual void SafeFree(void *p) { CUDA_ERROR(cudaFree(p)); }

	cudaStream_t stream_;
	DeviceMemoryInfo *dev_mem_;
	CudaContextProfiler cuda_context_profiler_;
};

// CacheCudaContext can mannually allocate a large portion of device memory
// and then keep malloc memory from that portion and release the whole portion
// of cache memory after the whole usage.
// This is useful to isolate the memory allocation overhead when testing the
// component, e.g., when testing performance of set intersection, and when
// process a batch during subgraph enumeration.
// CacheCudaContext is not needed as we have CnmemCudaContext.
// We keep it just because of some legacy code.
class CacheCudaContext : public CudaContext
{
public:
	CacheCudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
		: CudaContext(dev_mem, stream)
	{
		malloc_from_cache_ = false;
	}
	///////////////////////////////////////////////
	// general memory allocation using cuda API (expensive)
	virtual void *Malloc(size_t size)
	{
		cuda_context_profiler_.StartTimer(stream_);
		void *ret = NULL;
		if (!malloc_from_cache_)
		{
			ret = SafeMalloc(size);
			dev_mem_->Consume(size);
		}
		else
		{
			ret = MallocFromCache(size);
		}
		cuda_context_profiler_.EndTimer(stream_);
		return ret;
	}
	virtual void Free(void *p, size_t size)
	{
		cuda_context_profiler_.StartTimer(stream_);
		if (!malloc_from_cache_)
		{
			SafeFree(p);
			dev_mem_->Release(size);
		}
		cuda_context_profiler_.EndTimer(stream_);
	}

	// cache
	void SetMallocFromCache(bool f) { malloc_from_cache_ = f; }
	void MallocCache(size_t size)
	{
		assert(!malloc_from_cache_);
		void *base = Malloc(size);
		cache_alloc_.Init(base, size);
	}
	void FreeCache()
	{
		assert(!malloc_from_cache_);
		Free(cache_alloc_.GetBase(), cache_alloc_.GetSize());
		cache_alloc_.Reset();
	}

protected:
	void *MallocFromCache(size_t size) { return cache_alloc_.Malloc(size); }

	NoFreeCacheAllocator cache_alloc_;
	// if malloc_from_cache_=true, keep allocating from cache_alloc_
	// without releasing memory
	bool malloc_from_cache_;
};

#include "cnmem.h"

// CnmemCudaContext assumes CNMEM_FLAGS_MANAGED is not used.
// The allocation from cnmem allocator may reach the buffer limit and return
// CNMEM_STATUS_OUT_OF_MEMORY. In that case, we mannually call cudaMallocManaged
// to allocate from the unified memory. We assume that case is seldom, so
// cudaMallocManaged and cudaFree the requested memory if needed.
class CnmemCudaContext : public CudaContext
{
public:
	CnmemCudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
		: CudaContext(dev_mem, stream)
	{
		cnmem_device_ = new cnmemDevice_t();
		cnmem_device_->device = dev_mem_->GetDevId();
		cnmem_device_->size = dev_mem_->GetAvailableMemorySize();
		// In our use case, we have only one stream controls one device
		cnmem_device_->numStreams = 1;
		cnmem_device_->streams = &stream_;
		// Do not specify the memory reserved for each stream, so that the memory is
		// allocated when needed
		cnmem_device_->streamSizes = NULL;
	}
	~CnmemCudaContext()
	{
		delete cnmem_device_;
		cnmem_device_ = NULL;
	}

	cnmemDevice_t *GetCnmemDevice() const { return cnmem_device_; }

protected:
	static void CnmemAssertSuccess(cnmemStatus_t status)
	{
		if (status != CNMEM_STATUS_SUCCESS)
		{
			std::cerr << cnmemGetErrorString(status) << std::endl;
			assert(false);
		}
	}

	virtual void *DirectMalloc(size_t size)
	{
		void *ret = NULL;
		cnmemStatus_t status = cnmemMalloc(&ret, size, Stream());
		CnmemAssertSuccess(status);
		return ret;
	}
	virtual void DirectFree(void *p)
	{
		cnmemStatus_t status = cnmemFree(p, Stream());
		CnmemAssertSuccess(status);
	}
	virtual void *SafeMalloc(size_t size)
	{
		void *ret = NULL;
		int old_device;
		CUDA_ERROR(cudaGetDevice(&old_device));
		int new_device = dev_mem_->GetDevId();
		CUDA_ERROR(cudaSetDevice(new_device));
		cnmemStatus_t status = cnmemMalloc(&ret, size, Stream());
		if (status == CNMEM_STATUS_OUT_OF_MEMORY)
		{
			fprintf(stderr, "Insufficient device memory\n");
			// allocate from unified memory
			CUDA_ERROR(cudaMallocManaged(&ret, size));
			CUDA_ERROR(cudaMemPrefetchAsync(ret, size, dev_mem_->GetDevId()));
		}
		else
		{
			CnmemAssertSuccess(status);
		}
		CUDA_ERROR(cudaSetDevice(old_device));
		return ret;
	}
	virtual void SafeFree(void *p)
	{
		int old_device;
		CUDA_ERROR(cudaGetDevice(&old_device));
		int new_device = dev_mem_->GetDevId();
		CUDA_ERROR(cudaSetDevice(new_device));
		cnmemStatus_t status = cnmemFree(p, Stream());
		if (status == CNMEM_STATUS_INVALID_ARGUMENT)
		{
			// This pointer is not allocated from cnmem allocator
			CUDA_ERROR(cudaFree(p));
		}
		else
		{
			CnmemAssertSuccess(status);
		}
		CUDA_ERROR(cudaSetDevice(old_device));
	}

protected:
	cnmemDevice_t *cnmem_device_;
	size_t cnmem_stream_sizes_;
};

// CnmemManagedCudaContext assumes the flag is CNMEM_FLAGS_MANAGED and all the
// memory (including unified memory) is managed by the cnmem allocator. We do
// not need to worry about the case of large requested size.
class CnmemManagedCudaContext : public CnmemCudaContext
{
public:
	CnmemManagedCudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
		: CnmemCudaContext(dev_mem, stream) {}
	~CnmemManagedCudaContext() {}

protected:
	virtual void *DirectMalloc(size_t size)
	{
		if (!dev_mem_->IsAvailable(size))
		{
			fprintf(stderr, "Insufficient device memory\n");
		}
		void *ret = NULL;
		int old_device;
		CUDA_ERROR(cudaGetDevice(&old_device));
		int new_device = dev_mem_->GetDevId();
		CUDA_ERROR(cudaSetDevice(new_device));
		cnmemStatus_t status = cnmemMalloc(&ret, size, Stream());
		CnmemAssertSuccess(status);
		CUDA_ERROR(cudaSetDevice(old_device));
		return ret;
	}
	virtual void DirectFree(void *p)
	{
		int old_device;
		CUDA_ERROR(cudaGetDevice(&old_device));
		int new_device = dev_mem_->GetDevId();
		CUDA_ERROR(cudaSetDevice(new_device));
		cnmemStatus_t status = cnmemFree(p, Stream());
		CnmemAssertSuccess(status);
		CUDA_ERROR(cudaSetDevice(old_device));
	}

	virtual void *SafeMalloc(size_t size) { return DirectMalloc(size); }
	virtual void SafeFree(void *p) { DirectFree(p); }
};

#endif
