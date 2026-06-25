using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace YellowSquad.OceanLib
{
    public class Ocean : MonoBehaviour
    {
        [Header("Wave Spectrum")]
        [SerializeField] private float _windSpeed = 1f;
        [SerializeField] private Vector2 _windDirection = new Vector2(1f, 1f);
        [SerializeField] private float _gravity = 9.81f;
        [SerializeField] private float _fetch = 1f;
        [SerializeField] private float _depth = 4f;

        [Header("Rendering")]
        [SerializeField] private Material _material;
        [SerializeField] private int _texturesSize = 256;
        [SerializeField] private WaterCascade[] _cascades;

        [Header("Compute Shaders")]
        [SerializeField] private ComputeShader _initialSpectrumComputeShader;
        [SerializeField] private ComputeShader _timeDependentSpectrumComputeShader;
        [SerializeField] private ComputeShader _ifftComputeShader;
        [SerializeField] private ComputeShader _resultTexturesFillerComputeShader;
        [SerializeField] private ComputeShader _waterHeightSamplerComputeShader;

        private const int LocalWorkGroupsX = 8;
        private const int LocalWorkGroupsY = 8;

        private IFFT _ifft;
        private Texture2D _randomNoiseTexture;

        private RenderTexture _initialSpectrumTextures;
        private RenderTexture _wavesDataTextures;
        private RenderTexture _dxDzTextures;
        private RenderTexture _dyDxzTextures;
        private RenderTexture _dyxDyzTextures;
        private RenderTexture _dxxDzzTextures;
        private RenderTexture _displacementsTextures;
        private RenderTexture _derivativesTextures;
        private RenderTexture _turbulenceTextures;

        private float[] _wavelengths;
        private float[] _cutoffs;
        private float[] _swells;
        private float[] _fades;
        private ComputeBuffer _wavelengthsBuffer;
        private ComputeBuffer _cutoffsBuffer;
        private ComputeBuffer _swellsBuffer;
        private ComputeBuffer _fadesBuffer;

        private int _kernelInitialSpectrum;
        private int _kernelConjugatedSpectrum;
        private int _kernelTimeDependentSpectrum;
        private int _kernelResultTexturesFiller;

        private int _kernelSample4Heights;
        private ComputeBuffer _heightSamplePositionsBuffer;
        private ComputeBuffer _heightSampleOutBuffer;
        private readonly Vector3[] _queuedHeightSamplePositions = new Vector3[4];
        private bool _hasQueuedHeightSamples;
        private bool _heightReadbackInFlight;
        private bool _latestHeightsReady;
        private Vector4 _latestHeights;
        
        private void OnValidate()
        {
            if (_initialSpectrumTextures == null)
                return;

            _initialSpectrumComputeShader.SetFloat("_WindSpeed", _windSpeed);
            _initialSpectrumComputeShader.SetFloat("_WindDirectionX", _windDirection.x);
            _initialSpectrumComputeShader.SetFloat("_WindDirectionY", _windDirection.y);
            _initialSpectrumComputeShader.SetFloat("_Gravity", _gravity);
            _initialSpectrumComputeShader.SetFloat("_Fetch", _fetch);
            _initialSpectrumComputeShader.SetFloat("_Depth", _depth);
            CalculateInitialSpectrumTextures();
        }

        private void Awake()
        {
            GenerateRandomNoiseTexture();

            _ifft = new IFFT(_ifftComputeShader, _texturesSize, _cascades.Length);

            _kernelInitialSpectrum = _initialSpectrumComputeShader.FindKernel("CalculateInitialSpectrumTextures");
            _kernelConjugatedSpectrum = _initialSpectrumComputeShader.FindKernel("CalculateConjugatedInitialSpectrumTextures");
            _kernelTimeDependentSpectrum = _timeDependentSpectrumComputeShader.FindKernel("CalculateTimeDependentComplexAmplitudesAndDerivatives");
            _kernelResultTexturesFiller = _resultTexturesFillerComputeShader.FindKernel("FillResultTextures");

            _wavesDataTextures = CreateRenderTextureArray("wavesDataTextures", RenderTextureFormat.ARGBFloat);
            _initialSpectrumTextures = CreateRenderTextureArray("initialSpectrumTextures", RenderTextureFormat.ARGBFloat);
            _dxDzTextures = CreateRenderTextureArray("dxDzTextures", RenderTextureFormat.RGFloat);
            _dyDxzTextures = CreateRenderTextureArray("dyDxzTextures", RenderTextureFormat.RGFloat);
            _dyxDyzTextures = CreateRenderTextureArray("dyxDyzTextures", RenderTextureFormat.RGFloat);
            _dxxDzzTextures = CreateRenderTextureArray("dxxDzzTextures", RenderTextureFormat.RGFloat);
            _displacementsTextures = CreateRenderTextureArray("displacementsTextures", RenderTextureFormat.ARGBFloat);
            _derivativesTextures = CreateRenderTextureArray("derivativesTextures", RenderTextureFormat.ARGBFloat, useMips: true);
            _turbulenceTextures = CreateRenderTextureArray("turbulenceTextures", RenderTextureFormat.ARGBFloat, useMips: true);

            _wavelengths = new float[_cascades.Length];
            _cutoffs = new float[_cascades.Length * 2];
            _swells = new float[_cascades.Length];
            _fades = new float[_cascades.Length];

            for (int i = 0; i < _cascades.Length; i++)
            {
                _wavelengths[i] = _cascades[i].Wavelength;
                _cutoffs[i * 2] = _cascades[i].CutoffLow;
                _cutoffs[i * 2 + 1] = _cascades[i].CutoffHigh;
                _swells[i] = _cascades[i].Swell;
                _fades[i] = _cascades[i].Fade;
            }

            _wavelengthsBuffer = new ComputeBuffer(_cascades.Length, sizeof(float));
            _wavelengthsBuffer.SetData(_wavelengths);
            _cutoffsBuffer = new ComputeBuffer(_cascades.Length * 2, sizeof(float));
            _cutoffsBuffer.SetData(_cutoffs);
            _swellsBuffer = new ComputeBuffer(_cascades.Length, sizeof(float));
            _swellsBuffer.SetData(_swells);
            _fadesBuffer = new ComputeBuffer(_cascades.Length, sizeof(float));
            _fadesBuffer.SetData(_fades);

            InitializeInitialSpectrumComputeShader();
            CalculateInitialSpectrumTextures();
            InitializeTimeDependentSpectrumComputeShader();
            InitializeResultTexturesFillerComputeShader();

            if (_waterHeightSamplerComputeShader != null)
            {
                _kernelSample4Heights = _waterHeightSamplerComputeShader.FindKernel("Sample4Heights");
                _heightSamplePositionsBuffer = new ComputeBuffer(4, sizeof(float) * 4, ComputeBufferType.Structured);
                _heightSampleOutBuffer = new ComputeBuffer(4, sizeof(float), ComputeBufferType.Structured);
            }

            CreateRealtimeReflectionProbe();

            _material.SetInt("_NbCascades", _cascades.Length);
            _material.SetTexture("_DisplacementsTextures", _displacementsTextures);
            _material.SetTexture("_DerivativesTextures", _derivativesTextures);
            _material.SetTexture("_TurbulenceTextures", _turbulenceTextures);
            _material.SetFloatArray("_Wavelengths", _wavelengths);
        }
        
        private void OnDisable()
        {
            _wavelengthsBuffer?.Release();
            _cutoffsBuffer?.Release();
            _swellsBuffer?.Release();
            _fadesBuffer?.Release();
            _heightSamplePositionsBuffer?.Release();
            _heightSampleOutBuffer?.Release();

            _wavelengthsBuffer = null;
            _cutoffsBuffer = null;
            _swellsBuffer = null;
            _fadesBuffer = null;
            _heightSamplePositionsBuffer = null;
            _heightSampleOutBuffer = null;
        }

        private void Update()
        {
            CalculateWavesTexturesAtTime(Time.time);

            if (_waterHeightSamplerComputeShader != null && _hasQueuedHeightSamples && !_heightReadbackInFlight)
                DispatchHeightSamples();
        }

        public void QueueWaterHeightSamples4(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
        {
            _queuedHeightSamplePositions[0] = p0;
            _queuedHeightSamplePositions[1] = p1;
            _queuedHeightSamplePositions[2] = p2;
            _queuedHeightSamplePositions[3] = p3;
            _hasQueuedHeightSamples = true;
        }

        public bool TryGetLatestWaterHeights4(out Vector4 heights)
        {
            heights = _latestHeights;
            return _latestHeightsReady;
        }

        private void CalculateWavesTexturesAtTime(float time)
        {
            _timeDependentSpectrumComputeShader.SetFloat("_Time", time);
            _timeDependentSpectrumComputeShader.Dispatch(_kernelTimeDependentSpectrum, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);

            _ifft.InverseFastFourierTransform(_dxDzTextures);
            _ifft.InverseFastFourierTransform(_dyDxzTextures);
            _ifft.InverseFastFourierTransform(_dyxDyzTextures);
            _ifft.InverseFastFourierTransform(_dxxDzzTextures);

            _resultTexturesFillerComputeShader.Dispatch(_kernelResultTexturesFiller, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);

            _derivativesTextures.GenerateMips();
            _turbulenceTextures.GenerateMips();
        }

        private void DispatchHeightSamples()
        {
            var samplePositions = new Vector4[4];
            for (int i = 0; i < 4; i++)
            {
                var p = _queuedHeightSamplePositions[i];
                samplePositions[i] = new Vector4(p.x, p.y, p.z, 0f);
            }
            _heightSamplePositionsBuffer.SetData(samplePositions);

            _waterHeightSamplerComputeShader.SetInt("_NbCascades", _cascades.Length);
            _waterHeightSamplerComputeShader.SetBuffer(_kernelSample4Heights, "_Wavelengths", _wavelengthsBuffer);
            _waterHeightSamplerComputeShader.SetBuffer(_kernelSample4Heights, "_SamplePositions", _heightSamplePositionsBuffer);
            _waterHeightSamplerComputeShader.SetBuffer(_kernelSample4Heights, "_OutHeights", _heightSampleOutBuffer);
            _waterHeightSamplerComputeShader.SetTexture(_kernelSample4Heights, "_DisplacementsTextures", _displacementsTextures);

            _waterHeightSamplerComputeShader.Dispatch(_kernelSample4Heights, 1, 1, 1);

            _heightReadbackInFlight = true;
            _hasQueuedHeightSamples = false;

            AsyncGPUReadback.Request(_heightSampleOutBuffer, request =>
            {
                _heightReadbackInFlight = false;
                if (request.hasError)
                {
                    Debug.LogError($"[{nameof(Ocean)}] Height sample AsyncGPUReadback failed.");
                    _latestHeightsReady = false;
                    return;
                }

                NativeArray<float> data = request.GetData<float>();
                _latestHeights = new Vector4(data[0], data[1], data[2], data[3]);
                _latestHeightsReady = true;
            });
        }

        private static float GenerateRandomNumber()
        {
            float v1;
            float s;
            
            do
            {
                v1 = 2f * Random.value - 1f;
                float v2 = 2f * Random.value - 1f;
                s = v1 * v1 + v2 * v2;
            } while (s is >= 1f or 0f);

            return v1 * Mathf.Sqrt(-2f * Mathf.Log(s) / s);
        }

        private void GenerateRandomNoiseTexture()
        {
            var noiseTexture = new Texture2D(_texturesSize, _texturesSize, TextureFormat.RGFloat, false, true)
            {
                filterMode = FilterMode.Point
            };

            for (int i = 0; i < _texturesSize; i++)
            {
                for (int j = 0; j < _texturesSize; j++)
                {
                    noiseTexture.SetPixel(i, j, new Vector4(GenerateRandomNumber(), GenerateRandomNumber()));
                }
            }

            noiseTexture.Apply();
            _randomNoiseTexture = noiseTexture;
        }

        private RenderTexture CreateRenderTextureArray(string name, RenderTextureFormat format, bool useMips = false)
        {
            var renderTexture = new RenderTexture(_texturesSize, _texturesSize, 0, format, RenderTextureReadWrite.Linear)
            {
                name = name,
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = _cascades.Length,
                useMipMap = useMips,
                autoGenerateMips = false,
                anisoLevel = 16,
                filterMode = FilterMode.Trilinear,
                wrapMode = TextureWrapMode.Repeat,
                enableRandomWrite = true
            };
            renderTexture.Create();
            return renderTexture;
        }

        private void InitializeInitialSpectrumComputeShader()
        {
            _initialSpectrumComputeShader.SetInt("_TextureSize", _texturesSize);
            _initialSpectrumComputeShader.SetInt("_NbCascades", _cascades.Length);
            _initialSpectrumComputeShader.SetTexture(_kernelInitialSpectrum, "_RandomNoiseTexture", _randomNoiseTexture);
            _initialSpectrumComputeShader.SetTexture(_kernelInitialSpectrum, "_InitialSpectrumTextures", _initialSpectrumTextures);
            _initialSpectrumComputeShader.SetTexture(_kernelInitialSpectrum, "_WavesDataTextures", _wavesDataTextures);
            _initialSpectrumComputeShader.SetBuffer(_kernelInitialSpectrum, "_Wavelengths", _wavelengthsBuffer);
            _initialSpectrumComputeShader.SetBuffer(_kernelInitialSpectrum, "_Cutoffs", _cutoffsBuffer);
            _initialSpectrumComputeShader.SetBuffer(_kernelInitialSpectrum, "_Fades", _fadesBuffer);
            _initialSpectrumComputeShader.SetBuffer(_kernelInitialSpectrum, "_Swells", _swellsBuffer);
            _initialSpectrumComputeShader.SetFloat("_WindSpeed", _windSpeed);
            _initialSpectrumComputeShader.SetFloat("_WindDirectionX", _windDirection.x);
            _initialSpectrumComputeShader.SetFloat("_WindDirectionY", _windDirection.y);
            _initialSpectrumComputeShader.SetFloat("_Gravity", _gravity);
            _initialSpectrumComputeShader.SetFloat("_Fetch", _fetch);
            _initialSpectrumComputeShader.SetFloat("_Depth", _depth);

            _initialSpectrumComputeShader.SetTexture(_kernelConjugatedSpectrum, "_InitialSpectrumTextures", _initialSpectrumTextures);
        }

        private void InitializeTimeDependentSpectrumComputeShader()
        {
            _timeDependentSpectrumComputeShader.SetInt("_NbCascades", _cascades.Length);
            _timeDependentSpectrumComputeShader.SetTexture(_kernelTimeDependentSpectrum, "_ConjugatedInitialSpectrumTextures", _initialSpectrumTextures);
            _timeDependentSpectrumComputeShader.SetTexture(_kernelTimeDependentSpectrum, "_WavesDataTextures", _wavesDataTextures);
            _timeDependentSpectrumComputeShader.SetTexture(_kernelTimeDependentSpectrum, "_DxDzTextures", _dxDzTextures);
            _timeDependentSpectrumComputeShader.SetTexture(_kernelTimeDependentSpectrum, "_DyDxzTextures", _dyDxzTextures);
            _timeDependentSpectrumComputeShader.SetTexture(_kernelTimeDependentSpectrum, "_DyxDyzTextures", _dyxDyzTextures);
            _timeDependentSpectrumComputeShader.SetTexture(_kernelTimeDependentSpectrum, "_DxxDzzTextures", _dxxDzzTextures);
        }

        private void InitializeResultTexturesFillerComputeShader()
        {
            _resultTexturesFillerComputeShader.SetInt("_NbCascades", _cascades.Length);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_DxDzTextures", _dxDzTextures);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_DyDxzTextures", _dyDxzTextures);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_DyxDyzTextures", _dyxDyzTextures);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_DxxDzzTextures", _dxxDzzTextures);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_DisplacementsTextures", _displacementsTextures);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_DerivativesTextures", _derivativesTextures);
            _resultTexturesFillerComputeShader.SetTexture(_kernelResultTexturesFiller, "_TurbulenceTextures", _turbulenceTextures);
        }

        private void CalculateInitialSpectrumTextures()
        {
            _initialSpectrumComputeShader.Dispatch(_kernelInitialSpectrum, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);
            _initialSpectrumComputeShader.Dispatch(_kernelConjugatedSpectrum, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);
        }

        private void CreateRealtimeReflectionProbe()
        {
            var probeObject = new GameObject("RealtimeReflectionProbe");
            probeObject.transform.SetParent(transform);

            var reflectionProbe = probeObject.AddComponent<ReflectionProbe>();
            reflectionProbe.mode = ReflectionProbeMode.Realtime;
            reflectionProbe.refreshMode = ReflectionProbeRefreshMode.EveryFrame;
            reflectionProbe.timeSlicingMode = ReflectionProbeTimeSlicingMode.AllFacesAtOnce;
            reflectionProbe.clearFlags = ReflectionProbeClearFlags.Skybox;
            reflectionProbe.cullingMask = 0;

            var realtimeTexture = new RenderTexture(reflectionProbe.resolution, reflectionProbe.resolution, 16)
            {
                dimension = TextureDimension.Cube
            };
            realtimeTexture.Create();
            reflectionProbe.realtimeTexture = realtimeTexture;
        }
    }
}
