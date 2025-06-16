

make:
```
make clean && make
```
Debug:
add '-G'/'-g' to CFLAGS in Makefile
```
CFLAGS += -O3 [-G -g] -Xptxas=-v -std=c++17 -DUNTRACK_ALLOC_LARGE_VERTEX_NUM -DPTHREAD_LOCK \
					-DCUDA_CONTEXT_PROFILE
```
我的代码基于RPS的基础框架（例如内存分配等），依赖于moderngpu，cnmeme，cub库。可以从github下载并安装。我为了简便直接安装到库中避免每次修改LD_LIBRARY_PATH。

我们将顶点按照度数（进行了orientation后的度数）进行了分类，分为稠密点，中等点和稀疏点，可以用不同策略进行搜索。运行时可以指定不同的分类阈值和搜索时的chunk大小

run:
```
./triangle -f <datafile_path> -algo <algo_id(currently 10 is best)> -dense <threshold of dense vertices> -middle <threshold of middle vertices> -dc <search chunk of dense vertices> -mc <search chunk of middle vertices> -sc <search chunk of sparse vertices>
```

在data中有下载的脚本，目前我希望在graph500的graph500-scale24-ef16.bin的数据集下超过Mercury(用时500ms左右)，因为这个数据图是目前耗时相对较长的，可以看到明显提升。
