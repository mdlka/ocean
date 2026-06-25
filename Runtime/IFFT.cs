using UnityEngine;

namespace YellowSquad.OceanLib
{
    public class IFFT
    {
        private readonly int _texturesSize;

        private readonly ComputeShader _ifftComputeShader;
        private readonly RenderTexture _twiddleFactorsAndInputIndicesTexture;
        private readonly RenderTexture _pingPongTextures;

        private const int LocalWorkGroupsX = 8;
        private const int LocalWorkGroupsY = 8;

        private readonly int _kernelIfftHorizontalStep;
        private readonly int _kernelIfftVerticalStep;
        private readonly int _kernelIfftPermute;

        public IFFT(ComputeShader ifftComputeShader, int texturesSize, int nbCascades) 
        {
            _ifftComputeShader = ifftComputeShader;
            _texturesSize = texturesSize;

            int kernelIfftPrecomputeFactorsAndIndices = _ifftComputeShader.FindKernel("PrecomputeTwiddleFactorsAndInputIndices");
            _kernelIfftHorizontalStep = _ifftComputeShader.FindKernel("HorizontalStepIFFT");
            _kernelIfftVerticalStep = _ifftComputeShader.FindKernel("VerticalStepIFFT");
            _kernelIfftPermute = _ifftComputeShader.FindKernel("Permute");

            int logSize = (int)Mathf.Log(texturesSize, 2);
            _twiddleFactorsAndInputIndicesTexture = new RenderTexture(logSize, texturesSize, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.sRGB)
                {
                    name = "TwiddleFactorsAndInputIndicesTexture",
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Repeat,
                    enableRandomWrite = true
                };
            _twiddleFactorsAndInputIndicesTexture.Create();
            
            _ifftComputeShader.SetInt("_TextureSize", texturesSize);
            _ifftComputeShader.SetInt("_NbCascades", nbCascades);
            _ifftComputeShader.SetTexture(kernelIfftPrecomputeFactorsAndIndices, "_TwiddleFactorsAndInputIndicesTexture", _twiddleFactorsAndInputIndicesTexture);
            _ifftComputeShader.Dispatch(kernelIfftPrecomputeFactorsAndIndices, logSize, texturesSize/2/LocalWorkGroupsY, 1);

            _pingPongTextures = new RenderTexture(texturesSize, texturesSize, 0, RenderTextureFormat.RGFloat, RenderTextureReadWrite.Linear)
                {
                    name = "PingPongTextures",
                    dimension = UnityEngine.Rendering.TextureDimension.Tex2DArray,
                    volumeDepth = nbCascades,
                    useMipMap = false,
                    autoGenerateMips = false,
                    anisoLevel = 6,
                    filterMode = FilterMode.Trilinear,
                    wrapMode = TextureWrapMode.Repeat,
                    enableRandomWrite = true
                };
            _pingPongTextures.Create();
        }

        public void InverseFastFourierTransform(RenderTexture inputTexturesArray) 
        {
            int logSize = (int)Mathf.Log(_texturesSize, 2);
            bool pingPong = false;

            _ifftComputeShader.SetTexture(_kernelIfftHorizontalStep, "_TwiddleFactorsAndInputIndicesTexture", _twiddleFactorsAndInputIndicesTexture);
            _ifftComputeShader.SetTexture(_kernelIfftHorizontalStep, "_InputTextures", inputTexturesArray);
            _ifftComputeShader.SetTexture(_kernelIfftHorizontalStep, "_PingPongTextures", _pingPongTextures);

            for (int i = 0; i < logSize; i++) 
            {
                _ifftComputeShader.SetInt("_Step", i);
                _ifftComputeShader.SetBool("_PingPong", pingPong);
                _ifftComputeShader.Dispatch(_kernelIfftHorizontalStep, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);
                pingPong = !pingPong;
            }

            _ifftComputeShader.SetTexture(_kernelIfftVerticalStep, "_TwiddleFactorsAndInputIndicesTexture", _twiddleFactorsAndInputIndicesTexture);
            _ifftComputeShader.SetTexture(_kernelIfftVerticalStep, "_InputTextures", inputTexturesArray);
            _ifftComputeShader.SetTexture(_kernelIfftVerticalStep, "_PingPongTextures", _pingPongTextures);
            
            for (int i = 0; i < logSize; i++) 
            {
                _ifftComputeShader.SetInt("_Step", i);
                _ifftComputeShader.SetBool("_PingPong", pingPong);
                _ifftComputeShader.Dispatch(_kernelIfftVerticalStep, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);
                pingPong = !pingPong;
            }

            _ifftComputeShader.SetTexture(_kernelIfftPermute, "_InputTextures", inputTexturesArray);
            _ifftComputeShader.Dispatch(_kernelIfftPermute, _texturesSize / LocalWorkGroupsX, _texturesSize / LocalWorkGroupsY, 1);
        }
    }
}
