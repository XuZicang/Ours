#include "Meta.h"
#include "utils/DeviceArray.cuh"
class Stack
{
    DeviceArray<uint32_t>* stacks_;
    uint32_t warpNum;
    uint32_t maxDeg;
    uint32_t layerNum;
    CudaContext* context_;
public:
    Stack(uint32_t _warpNum, uint32_t _maxDeg, uint32_t _layerNum, CudaContext* context): warpNum(_warpNum), maxDeg(_maxDeg), layerNum(_layerNum), context_(context) 
    {
        stacks_ = NULL;
    }
    uint32_t* allocate_binary_stacks()
    {
        if (stacks_ != NULL)
            delete stacks_;
        stacks_ = new DeviceArray<uint32_t>(warpNum * layerNum * ((maxDeg + 1023) / 1024) * 32, context_);
        return stacks_->GetArray();
    }
    uintV* allocate_stacks()
    {
        if (stacks_ != NULL)
            delete stacks_;
        stacks_ = new DeviceArray<uint32_t>(warpNum * layerNum * ((maxDeg + 31) / 32) * 32, context_);
        return reinterpret_cast<uintV*>(stacks_->GetArray());
    }
    ~Stack() {
        if (stacks_ != NULL)
            delete stacks_;
    }
};