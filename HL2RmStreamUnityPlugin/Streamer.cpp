#include "pch.h"

#define DBG_ENABLE_INFO_LOGGING 1
#define DBG_ENABLE_ERROR_LOGGING 1
#define DBG_ENABLE_VERBOSE_LOGGING 0

using namespace winrt::Windows::Networking::Sockets;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Perception;
using namespace winrt::Windows::Perception::Spatial;
using namespace winrt::Windows::Foundation::Numerics;

Streamer::Streamer(
    std::wstring portName,
    const GUID& guid,
    const winrt::Windows::Perception::Spatial::SpatialCoordinateSystem& coordSystem)
{
    m_portName = portName;
    m_worldCoordSystem = coordSystem;

    // Get GUID identifying the rigNode to
    // initialize the SpatialLocator
    SetLocator(guid);

    StartServer();
}

winrt::Windows::Foundation::IAsyncAction Streamer::StartServer()
{
    try
    {
        // The ConnectionReceived event is raised when connections are received.
        m_streamSocketListener.ConnectionReceived({ this, &Streamer::OnConnectionReceived });

        // Start listening for incoming TCP connections on the specified port. You can specify any port that's not currently in use.
        // Every protocol typically has a standard port number. For example, HTTP is typically 80, FTP is 20 and 21, etc.
        // For this example, we'll choose an arbitrary port number.
        co_await m_streamSocketListener.BindServiceNameAsync(m_portName);
#if DBG_ENABLE_INFO_LOGGING
        wchar_t msgBuffer[200];
        swprintf_s(msgBuffer, L"Streamer::StartServer: Server is listening at %ls. \n",
            m_portName.c_str());
        OutputDebugStringW(msgBuffer);
#endif // DBG_ENABLE_INFO_LOGGING

    }
    catch (winrt::hresult_error const& ex)
    {
#if DBG_ENABLE_ERROR_LOGGING
        SocketErrorStatus webErrorStatus{ SocketError::GetStatus(ex.to_abi()) };
        winrt::hstring message = webErrorStatus != SocketErrorStatus::Unknown ?
            winrt::to_hstring((int32_t)webErrorStatus) : winrt::to_hstring(ex.to_abi());
        OutputDebugStringW(L"Streamer::StartServer: Failed to open listener with ");
        OutputDebugStringW(message.c_str());
        OutputDebugStringW(L"\n");
#endif
    }
}

void Streamer::OnConnectionReceived(
    StreamSocketListener /* sender */,
    StreamSocketListenerConnectionReceivedEventArgs args)
{
    m_streamSocket = args.Socket();
    m_writer = args.Socket().OutputStream();
    m_writer.UnicodeEncoding(UnicodeEncoding::Utf8);
    m_writer.ByteOrder(ByteOrder::LittleEndian);

    m_writeInProgress = false;
    m_streamingEnabled = true;
#if DBG_ENABLE_INFO_LOGGING
    wchar_t msgBuffer[200];
    swprintf_s(msgBuffer, L"Streamer::OnConnectionReceived: Received connection at %ls. \n",
        m_portName.c_str());
    OutputDebugStringW(msgBuffer);
#endif // DBG_ENABLE_INFO_LOGGING
}

bool Streamer::SendExtrinsics(
    IResearchModeSensorFrame* frame,
    IResearchModeSensor* pSensor)
{
#if DBG_ENABLE_INFO_LOGGING
    OutputDebugStringW(L"Streamer::SendExtrinsics: Received frame for sending!\n");
#endif

    bool b_successful_write = false;

    if (!m_streamSocket || !m_writer)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendExtrinsics: No connection.\n");
#endif
        return b_successful_write;
    }
    if (!m_streamingEnabled)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendExtrinsics: Streaming disabled.\n");
#endif
        return b_successful_write;
    }

    // grab the frame info
    ResearchModeSensorTimestamp rmTimestamp;
    winrt::check_hresult(frame->GetTimeStamp(&rmTimestamp));
    auto prevTimestamp = rmTimestamp.HostTicks;

    auto timestamp = PerceptionTimestampHelper::FromSystemRelativeTargetTime(HundredsOfNanoseconds(checkAndConvertUnsigned(prevTimestamp)));
    auto location = m_locator.TryLocateAtTimestamp(timestamp, m_worldCoordSystem);
    if (!location)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendExtrinsics: Can't locate frame.\n");
#endif
        return b_successful_write;
    }

    ResearchModeSensorResolution resolution;
    size_t outBufferCount;
    frame->GetResolution(&resolution);

    int imageWidth = resolution.Width;
    int imageHeight = resolution.Height;

    ResearchModeSensorType pSensorType = pSensor->GetSensorType();

    if (pSensorType == DEPTH_AHAT)
    {
        IResearchModeSensorDepthFrame* pDepthFrame = nullptr;
        HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&pDepthFrame));

        if (!pDepthFrame)
        {
#if DBG_ENABLE_VERBOSE_LOGGING
            OutputDebugStringW(L"Streamer::SendFrame: Failed to grab depth frame.\n");
#endif
            return b_successful_write;
        }

        if (pDepthFrame)
        {
            pDepthFrame->Release();
        }
    }

    if (pSensorType == LEFT_FRONT)
    {
        IResearchModeSensorVLCFrame* pLFFrame = nullptr;
        HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&pLFFrame));

        if (!pLFFrame)
        {
#if DBG_ENABLE_VERBOSE_LOGGING
            OutputDebugStringW(L"Streamer::SendFrame: Failed to grab front left frame.\n");
#endif
            return b_successful_write;
        }

        // release space
        if (pLFFrame)
        {
            pLFFrame->Release();
        }
    }

    if (pSensorType == RIGHT_FRONT)
    {
        IResearchModeSensorVLCFrame* pRFFrame = nullptr;
        HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&pRFFrame));

        if (!pRFFrame)
        {
#if DBG_ENABLE_VERBOSE_LOGGING
            OutputDebugStringW(L"Streamer::SendFrame: Failed to grab front left frame.\n");
#endif
            return b_successful_write;
        }

        // release space
        if (pRFFrame)
        {
            pRFFrame->Release();
        }
    }

    // Get camera sensor object
    IResearchModeCameraSensor* pCameraSensor = nullptr;
    HRESULT hr = pSensor->QueryInterface(IID_PPV_ARGS(&pCameraSensor));
    winrt::check_hresult(hr);

    if (!pCameraSensor)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendExtrinsics: Failed to grab sensor for extrinsics.\n");
#endif
        return b_successful_write;
    }

    // Get extrinsics (rotation and translation) with respect to the rigNode
    DirectX::XMFLOAT4X4 cameraViewMatrix;
    pCameraSensor->GetCameraExtrinsicsMatrix(&cameraViewMatrix);

    const float4x4 sensorExtrinsics(
        cameraViewMatrix.m[0][0], cameraViewMatrix.m[1][0], cameraViewMatrix.m[2][0], cameraViewMatrix.m[3][0],
        cameraViewMatrix.m[0][1], cameraViewMatrix.m[1][1], cameraViewMatrix.m[2][1], cameraViewMatrix.m[3][1],
        cameraViewMatrix.m[0][2], cameraViewMatrix.m[1][2], cameraViewMatrix.m[2][2], cameraViewMatrix.m[3][2],
        cameraViewMatrix.m[0][3], cameraViewMatrix.m[1][3], cameraViewMatrix.m[2][3], cameraViewMatrix.m[3][3]);

    float uv[2];
    float xy[2];
    std::vector<float> lutTable(size_t(imageWidth * imageHeight) * 3);
    auto pLutTable = lutTable.data();

    // todo: we only want to send this extrinsics data once, not in send function
    for (size_t y = 0; y < resolution.Height; y++)
    {
        uv[1] = (y + 0.5f);
        for (size_t x = 0; x < resolution.Width; x++)
        {
            uv[0] = (x + 0.5f);
            hr = pCameraSensor->MapImagePointToCameraUnitPlane(uv, xy);
            if (FAILED(hr))
            {
                *pLutTable++ = xy[0];
                *pLutTable++ = xy[1];
                *pLutTable++ = 0.f;
                continue;
            }
            float z = 1.0f;
            const float norm = sqrtf(xy[0] * xy[0] + xy[1] * xy[1] + z * z);
            const float invNorm = 1.0f / norm;
            xy[0] *= invNorm;
            xy[1] *= invNorm;
            z *= invNorm;

            // Dump LUT row
            *pLutTable++ = xy[0];
            *pLutTable++ = xy[1];
            *pLutTable++ = z;
        }
    }

    pCameraSensor->Release();

    const unsigned char* lutToBytes = reinterpret_cast<const unsigned char*> (lutTable.data());
    std::vector<unsigned char> lutTableByteData(lutToBytes, lutToBytes + sizeof(float) * lutTable.size());

    if (m_writeInProgress)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendExtrinsics: Write already in progress.\n");
#endif
        return b_successful_write;
    }

    m_writeInProgress = true;

    try
    {
        // Write header
        WriteMatrix4x4(sensorExtrinsics);
        m_writer.WriteBytes(lutTableByteData);

#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendExtrinsics: Trying to store writer...\n");
#endif
        m_writer.StoreAsync();

        b_successful_write = true;
    }
    catch (winrt::hresult_error const& ex)
    {
        SocketErrorStatus webErrorStatus{ SocketError::GetStatus(ex.to_abi()) };
        if (webErrorStatus == SocketErrorStatus::ConnectionResetByPeer)
        {
            // the client disconnected!
            m_writer == nullptr;
            m_streamSocket == nullptr;
            m_writeInProgress = false;
        }
#if DBG_ENABLE_ERROR_LOGGING
        winrt::hstring message = ex.message();
        OutputDebugStringW(L"Streamer::SendExtrinsics: Sending failed with ");
        OutputDebugStringW(message.c_str());
        OutputDebugStringW(L"\n");
#endif // DBG_ENABLE_ERROR_LOGGING
    }

    m_writeInProgress = false;

#if DBG_ENABLE_VERBOSE_LOGGING
    OutputDebugStringW(L"Streamer::SendExtrinsics: Extrinsics sent!\n");
#endif
    
    return b_successful_write;
}

void Streamer::Send(
    IResearchModeSensorFrame* frame,
    ResearchModeSensorType pSensorType)
{
#if DBG_ENABLE_INFO_LOGGING
    OutputDebugStringW(L"Streamer::Send: Received frame for sending!\n");
#endif

    if (!m_streamSocket || !m_writer)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendFrame: No connection.\n");
#endif
        return;
    }
    if (!m_streamingEnabled)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendFrame: Streaming disabled.\n");
#endif
        return;
    }

    // grab the frame info
    ResearchModeSensorTimestamp rmTimestamp;
    winrt::check_hresult(frame->GetTimeStamp(&rmTimestamp));
    auto prevTimestamp = rmTimestamp.HostTicks;

    auto timestamp = PerceptionTimestampHelper::FromSystemRelativeTargetTime(HundredsOfNanoseconds(checkAndConvertUnsigned(prevTimestamp)));
    auto location = m_locator.TryLocateAtTimestamp(timestamp, m_worldCoordSystem);
    if (!location)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendFrame: Can't locate frame.\n");
#endif
        return;
    }
    const float4x4 rig2worldTransform = make_float4x4_from_quaternion(location.Orientation()) * make_float4x4_translation(location.Position());
    auto absoluteTimestamp = m_converter.RelativeTicksToAbsoluteTicks(HundredsOfNanoseconds((long long)prevTimestamp)).count();

    ResearchModeSensorResolution resolution;
    size_t outBufferCount;
    frame->GetResolution(&resolution);

    int imageWidth = resolution.Width;
    int imageHeight = resolution.Height;
    int pixelStride = resolution.BytesPerPixel;

    int rowStride = imageWidth * pixelStride;
    
    // grab the frame data
    std::vector<BYTE> sensorByteData;
    
    if (pSensorType == DEPTH_AHAT)
    {        
        IResearchModeSensorDepthFrame* pDepthFrame = nullptr;
        
        const UINT16* pDepth = nullptr;

        // invalidation value for AHAT 
        USHORT maxValue = 4090;

        HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&pDepthFrame));

        if (!pDepthFrame)
        {
#if DBG_ENABLE_VERBOSE_LOGGING
            OutputDebugStringW(L"Streamer::SendFrame: Failed to grab depth frame.\n");
#endif
            return;
        }

        hr = pDepthFrame->GetBuffer(&pDepth, &outBufferCount);
        sensorByteData.reserve(outBufferCount * sizeof(UINT16));

        // validate depth & append to vector
        for (size_t i = 0; i < outBufferCount; ++i)
        {
            // use a different invalidation condition for Long Throw and AHAT 
            const bool invalid = (pDepth[i] >= maxValue);
            UINT16 d;
            if (invalid)
            {
                d = 0;
            }
            else
            {
                d = pDepth[i];
            }
            sensorByteData.push_back((BYTE)(d >> 8));
            sensorByteData.push_back((BYTE)d);
        }

        if (pDepthFrame)
        {
            pDepthFrame->Release();
        }
    }

    if (pSensorType == LEFT_FRONT)
    {
        IResearchModeSensorVLCFrame* pLFFrame = nullptr;
        HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&pLFFrame));

        if (!pLFFrame)
        {
#if DBG_ENABLE_VERBOSE_LOGGING
            OutputDebugStringW(L"Streamer::SendFrame: Failed to grab front left frame.\n");
#endif
            return;
        }

        size_t outBufferCount = 0;
        const BYTE* pLFImage = nullptr;

        hr = pLFFrame->GetBuffer(&pLFImage, &outBufferCount);
        sensorByteData.reserve(outBufferCount * sizeof(UINT8));

        // validate depth & append to vector
        for (size_t i = 0; i < outBufferCount; ++i)
        {
            BYTE d = pLFImage[i];
            sensorByteData.push_back(d);
        }

        // release space
        if (pLFFrame)
        {
            pLFFrame->Release();
        }
    }    
    
    if (pSensorType == RIGHT_FRONT)
    {
        IResearchModeSensorVLCFrame* pRFFrame = nullptr;
        HRESULT hr = frame->QueryInterface(IID_PPV_ARGS(&pRFFrame));

        if (!pRFFrame)
        {
#if DBG_ENABLE_VERBOSE_LOGGING
            OutputDebugStringW(L"Streamer::SendFrame: Failed to grab front left frame.\n");
#endif
            return;
        }

        size_t outBufferCount = 0;
        const BYTE* pRFImage = nullptr;

        hr = pRFFrame->GetBuffer(&pRFImage, &outBufferCount);
        sensorByteData.reserve(outBufferCount * sizeof(UINT8));

        // validate depth & append to vector
        for (size_t i = 0; i < outBufferCount; ++i)
        {
            BYTE d = pRFImage[i];
            sensorByteData.push_back(d);
        }

        // release space
        if (pRFFrame)
        {
            pRFFrame->Release();
        }
    }

    if (m_writeInProgress)
    {
#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendFrame: Write already in progress.\n");
#endif
        return;
    }

    m_writeInProgress = true;

    try
    {
        // Write header
        m_writer.WriteUInt64(absoluteTimestamp);
        m_writer.WriteInt32(imageWidth);
        m_writer.WriteInt32(imageHeight);
        m_writer.WriteInt32(pixelStride);
        m_writer.WriteInt32(rowStride);

        WriteMatrix4x4(rig2worldTransform);

        m_writer.WriteBytes(sensorByteData);

#if DBG_ENABLE_VERBOSE_LOGGING
        OutputDebugStringW(L"Streamer::SendFrame: Trying to store writer...\n");
#endif
        m_writer.StoreAsync();
    }
    catch (winrt::hresult_error const& ex)
    {
        SocketErrorStatus webErrorStatus{ SocketError::GetStatus(ex.to_abi()) };
        if (webErrorStatus == SocketErrorStatus::ConnectionResetByPeer)
        {
            // the client disconnected!
            m_writer == nullptr;
            m_streamSocket == nullptr;
            m_writeInProgress = false;
        }
#if DBG_ENABLE_ERROR_LOGGING
        winrt::hstring message = ex.message();
        OutputDebugStringW(L"Streamer::SendFrame: Sending failed with ");
        OutputDebugStringW(message.c_str());
        OutputDebugStringW(L"\n");
#endif // DBG_ENABLE_ERROR_LOGGING
    }

    m_writeInProgress = false;

#if DBG_ENABLE_VERBOSE_LOGGING
    OutputDebugStringW(L"Streamer::SendFrame: Frame sent!\n");
#endif
}

void Streamer::StreamingToggle()
{
#if DBG_ENABLE_INFO_LOGGING
    OutputDebugStringW(L"Streamer::StreamingToggle: Received!\n");
#endif
    if (m_streamingEnabled)
    {
        m_streamingEnabled = false;
    }
    else if (!m_streamingEnabled)
    {
        m_streamingEnabled = true;
    }
#if DBG_ENABLE_INFO_LOGGING
    OutputDebugStringW(L"Streamer::StreamingToggle: Done!\n");
#endif
}

void Streamer::WriteMatrix4x4(
    _In_ winrt::Windows::Foundation::Numerics::float4x4 matrix)
{
    m_writer.WriteSingle(matrix.m11);
    m_writer.WriteSingle(matrix.m12);
    m_writer.WriteSingle(matrix.m13);
    m_writer.WriteSingle(matrix.m14);

    m_writer.WriteSingle(matrix.m21);
    m_writer.WriteSingle(matrix.m22);
    m_writer.WriteSingle(matrix.m23);
    m_writer.WriteSingle(matrix.m24);

    m_writer.WriteSingle(matrix.m31);
    m_writer.WriteSingle(matrix.m32);
    m_writer.WriteSingle(matrix.m33);
    m_writer.WriteSingle(matrix.m34);

    m_writer.WriteSingle(matrix.m41);
    m_writer.WriteSingle(matrix.m42);
    m_writer.WriteSingle(matrix.m43);
    m_writer.WriteSingle(matrix.m44);
}

void Streamer::SetLocator(const GUID& guid)
{
    m_locator = Preview::SpatialGraphInteropPreview::CreateLocatorForNode(guid);
}
