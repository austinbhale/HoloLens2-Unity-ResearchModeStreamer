#pragma once

class IResearchModeFrameSink
{
public:
	virtual ~IResearchModeFrameSink() {};
	virtual void Send(
		IResearchModeSensorFrame* pSensorFrame,
		IResearchModeSensor* pSensor) = 0;
}; 
