using System;
using UnityEngine;

namespace YellowSquad.OceanLib
{
    public class WaterCascade : MonoBehaviour
    {
        [SerializeField] private float _wavelength = 10f;
        [SerializeField] private float _cutoffHigh = 5f;
        [SerializeField] private float _cutoffLow = 0.0001f;
        [SerializeField] private float _swell = 0.4f;
        [SerializeField] private float _fade = 0.1f;

        public float Wavelength => _wavelength;
        public float CutoffHigh => _cutoffHigh;
        public float CutoffLow => _cutoffLow;
        public float Swell => _swell;
        public float Fade => _fade;
    }
}
