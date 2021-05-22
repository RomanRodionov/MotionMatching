#pragma once
#include "Serialization/settings_set.h"
struct Settings : SettingsSet
{
  static inline Settings *instance;
  #define FVAR DECL_FVAR
  #define IVAR DECL_IVAR
  #define BVAR DECL_BVAR
  #define LABEL(s)

  #define PARAMS()\
  FVAR(walkForwardSpeed, 1.2f, 0.f, 20.f)\
  FVAR(walkSidewaySpeed, 1.2f, 0.f, 20.f)\
  FVAR(walkBackwardSpeed, 1.0f, 0.f, 20.f)\
  FVAR(runForwardSpeed, 2.7f*1.2f, 0.f, 20.f)\
  FVAR(runSidewaySpeed, 2.f*1.2f, 0.f, 20.f)\
  FVAR(runBackwardSpeed, 2.1f*1.2f, 0.f, 20.f)\
  FVAR(hipsHeightStand, 0.967f, 0.f, 20.f)\
  FVAR(hipsHeightCrouch, 0.35f, 0.f, 20.f)\
  FVAR(hipsHeightJump, 1.2f, 0.f, 20.f)\
  FVAR(predictionMoveRate, 9.0f, 0.f, 20.f)\
  FVAR(predictionRotationRate, 9.0f, 0.f, 20.f)\
  FVAR(rotationRate, 2.0f, 0.f, 20.f)\
  FVAR(maxMoveErrorRadius, 1.0f, 0.f, 5.f)\
  FVAR(mouseSensitivity, 0.2f, 0.f, 1.f)\
  BVAR(mouseInvertXaxis, true)\
  FVAR(lerpTime, 0.2f, 0.f, 1.f)\
  IVAR(maxLerpIndex, 2, 2, 10)\
  BVAR(debugNodes, false)\
  BVAR(debugTrajectory, false)\
  BVAR(debugBones, false)\
  BVAR(MatchingStatistic, false)

  PARAMS()
  #undef FVAR
  #undef IVAR
  #undef BVAR
  #undef LABEL


  Settings()
  {
    #define FVAR INIT_FVAR
    #define IVAR INIT_IVAR
    #define BVAR INIT_BVAR
    #define LABEL INIT_LABEL
    PARAMS() 
    #undef PARAMS
    #undef FVAR
    #undef IVAR
    #undef BVAR
    #undef LABEL
  }


};