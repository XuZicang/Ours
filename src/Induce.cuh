#pragma once
#include "Meta.h"
#include <cuda.h>
#include "utils/DeviceArray.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;


class InduceGraph
{
    uint32_t blockNum;
    uint32_t maxDeg;
    CudaContext* context_;
    DeviceArray<uint32_t>* Gind_;
    DeviceArray<uint32_t>* Vorient_;
    DeviceArray<uintV> *Vind_;
public:
    InduceGraph(uint32_t _blockNum, uint32_t _maxDeg, CudaContext* context): blockNum(_blockNum), maxDeg(_maxDeg), context_(context)
    {
        Gind_ = NULL;
        Vorient_ = NULL;
        Vind_ = NULL;
    }
    uint32_t* allocate_Gind()
    {
        if (Gind_ != NULL) delete Gind_;
        Gind_ = new DeviceArray<uint32_t>(blockNum * maxDeg * ((maxDeg + 1023) / 1024) * 32, context_);
        return Gind_->GetArray();
    }
    uint32_t* allocate_Vorient()
    {
        if (Vorient_ != NULL) delete Vorient_;
        Vorient_ = new DeviceArray<uint32_t> (blockNum * ((maxDeg + 1023) / 1024) * 32, context_);
        return Vorient_->GetArray();
    }
    uintV* allocate_Vind()
    {
        if (Vind_ != NULL) delete Vind_;
        Vind_ = new DeviceArray<uintV> (blockNum * (maxDeg + 31) / 32 * 32, context_);
        return Vind_->GetArray();
    }
    ~InduceGraph() {
        if (Gind_ != NULL)    delete Gind_;
        Gind_ = NULL;
        if (Vorient_ != NULL) delete Vorient_;
        Vorient_ = NULL;
        if (Vind_ != NULL) delete Vind_;
        Vind_ = NULL;
    }
};