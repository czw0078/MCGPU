#ifndef GPUCOPY_H
#define GPUCOPY_H

#include "SimBox.h"

namespace GPUCopy {

  void setParallel(bool);
  bool onGpu();
  void setParallel(bool launchOnGpu);

  void copyIn(SimBox* sb);
  void copyOut(SimBox* sb);

  SimBox* simBoxGPU();
  SimBox* simBoxCPU();

  Real* atomCoordinatesPtr();
  int* moleculeDataPtr();

  Real** angleDataPtr();
  Real* angleSizesPtr();
  Real* rollBackAngleSizesPtr();

  Real** bondDataPtr();
  Real* bondLengthsPtr();
  Real* rollBackBondLengthsPtr();
}

#endif
