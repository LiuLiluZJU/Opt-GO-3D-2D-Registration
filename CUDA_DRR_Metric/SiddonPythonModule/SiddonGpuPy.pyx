import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C++ code
cdef extern from "include/siddon_class.cuh" :
    cdef cppclass SiddonGpu :
        SiddonGpu()
        SiddonGpu(int *NumThreadsPerBlock,
                  float *movImgArray,
                  int *MovSize,
                  float *MovSpacing,
                  float X0, float Y0, float Z0,
                  int *DRRSize)
        void generateDRR(float *source,
						float *DestArray,
						float *drrArray)
        void computeMetricMedian(float *fixedArray,
						float *drrGradientMap,
						float *fixedGradientMap)
        void computeMetric(float *fixedArray,
						float drrThreshold,
						float fixedThreshold,
						int lowNum,
						float *weightSum,
						int *weightNum,
						float *metricValue)
        void backwardProp(float *source,
						float *DestArray,
						float *inputGradArray, 
						float *outputGradArray)

cdef class pySiddonGpu :
    cdef SiddonGpu* thisptr # hold a C++ instance
    cdef DRRsize
    cdef MovSize
    def __cinit__(self, np.ndarray[int, ndim = 1, mode = "c"] NumThreadsPerBlock not None,
                        np.ndarray[float, ndim = 1, mode = "c"] movImgArray not None,
                        np.ndarray[int, ndim = 1, mode = "c"] MovSize not None,
                        np.ndarray[float, ndim = 1, mode = "c"] MovSpacing not None,
                        X0, Y0, Z0,
                        np.ndarray[int, ndim = 1, mode = "c"] DRRsize not None) :

        self.DRRsize = DRRsize
        self.MovSize = MovSize
        self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],
                                        &movImgArray[0],
                                        &MovSize[0],
                                        &MovSpacing[0],
                                        X0, Y0, Z0,
                                        &DRRsize[0])

    def generateDRR(self, np.ndarray[float, ndim = 1, mode = "c"] source not None,
                          np.ndarray[float, ndim = 1, mode = "c"] DestArray not None) :

        # generate contiguous output array
        drrArray = np.zeros(self.DRRsize[0] * self.DRRsize[1] * self.DRRsize[2], dtype = np.float32, order = 'C')
        # weightSum = np.ndarray([0], dtype=np.float32, order = 'C')
        # weightNum = np.ndarray([0], dtype=np.int32, order = 'C')
        cdef float[::1] cdrrArray = drrArray
        # cdef np.ndarray[float, ndim=1] cweightSum = weightSum
        # cdef np.ndarray[int, ndim=1] cweightNum = weightNum

        self.thisptr.generateDRR(&source[0], &DestArray[0], &cdrrArray[0])

        return cdrrArray

    def computeMetricMedian(self, np.ndarray[float, ndim = 1, mode = "c"] fixedArray not None):

        # generate contiguous output array
        drrGradientMapArray = np.zeros((self.DRRsize[0] - 1) * (self.DRRsize[1] - 1) * self.DRRsize[2], dtype = np.float32, order = 'C')
        fixedGradientMapArray = np.zeros((self.DRRsize[0] - 1) * (self.DRRsize[1] - 1) * self.DRRsize[2], dtype = np.float32, order = 'C')

        cdef float[::1] cdrrGradientMapArray = drrGradientMapArray
        cdef float[::1] cfixedGradientMapArray = fixedGradientMapArray

        self.thisptr.computeMetricMedian(&fixedArray[0], &cdrrGradientMapArray[0], &cfixedGradientMapArray[0])

        return [cdrrGradientMapArray, cfixedGradientMapArray]

    def computeMetric(self, np.ndarray[float, ndim = 1, mode = "c"] fixedArray not None,
                            drrThreshold = 100,
                            fixedThreshold = 100,
                            lowNum = 500) :
        cdef float cweightSum = 0
        cdef int cweightNum = 0
        cdef float cmetricValue = 0

        self.thisptr.computeMetric(&fixedArray[0], 
                                    drrThreshold, 
                                    fixedThreshold, 
                                    lowNum, 
                                    &cweightSum, 
                                    &cweightNum,
                                    &cmetricValue)

        return [cweightSum, cweightNum, cmetricValue]

    def backwardProp(self, np.ndarray[float, ndim = 1, mode = "c"] source not None,
                           np.ndarray[float, ndim = 1, mode = "c"] DestArray not None,
                           np.ndarray[float, ndim = 1, mode = "c"] inputGradArray not None) :
        outputGradArray = np.zeros(self.MovSize[0] * self.MovSize[1] * self.MovSize[2], dtype = np.float32, order = 'C')
        cdef float[::1] coutputGradArray = outputGradArray
        self.thisptr.backwardProp(&source[0], &DestArray[0], &inputGradArray[0], &coutputGradArray[0])

        return coutputGradArray

    def delete(self) :
        if self.thisptr is not NULL :
            "C++ object being destroyed"
            del self.thisptr

