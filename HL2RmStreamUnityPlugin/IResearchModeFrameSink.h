#pragma once

class IResearchModeFrameSink
{
public:
	virtual ~IResearchModeFrameSink() {};
	virtual bool SendExtrinsics(
		IResearchModeSensorFrame* pSensorFrame,
		IResearchModeSensor* pSensor) = 0;	
	virtual void Send(
		IResearchModeSensorFrame* pSensorFrame,
		ResearchModeSensorType pSensorType) = 0;
}; 
