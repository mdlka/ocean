using UnityEngine;

namespace YellowSquad.OceanLib
{
    [RequireComponent(typeof(Rigidbody))]
    public class FloatingBody : MonoBehaviour
    {
        [SerializeField] private Ocean _ocean;
        [SerializeField] private float _sampleRightOffset = 0.5f;
        [SerializeField] private float _sampleForwardOffset = 1.0f;
        [SerializeField] private float _pivotOffsetY = 0.0f;
        [SerializeField] private float _positionLerp = 15f;
        [SerializeField] private float _rotationLerp = 10f;

        private Rigidbody _rigidbody;
        private readonly Vector3[] _samplePoints = new Vector3[4];

        private void Start()
        {
            _rigidbody = GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            if (_ocean == null)
                return;

            UpdateSamplePoints();
            _ocean.QueueWaterHeightSamples4(_samplePoints[0], _samplePoints[1], _samplePoints[2], _samplePoints[3]);

            if (!_ocean.TryGetLatestWaterHeights4(out Vector4 heights))
                return;

            AlignRotationToSurface(heights);
            AlignPositionToSurface(heights);
        }

        private void OnDrawGizmos()
        {
            if (_ocean == null || !_ocean.TryGetLatestWaterHeights4(out Vector4 heights))
                return;

            UpdateSamplePoints();
            for (int i = 0; i < _samplePoints.Length; i++)
            {
                Vector3 p = _samplePoints[i];
                p.y = heights[i];
                Gizmos.DrawSphere(p, 0.15f);
            }
        }

        private void UpdateSamplePoints()
        {
            Vector3 center = transform.position;
            Vector3 right = transform.right * _sampleRightOffset;
            Vector3 forward = transform.forward * _sampleForwardOffset;

            _samplePoints[0] = center + right + forward;
            _samplePoints[1] = center - right + forward;
            _samplePoints[2] = center - right - forward;
            _samplePoints[3] = center + right - forward;
        }

        private void AlignRotationToSurface(Vector4 heights)
        {
            var a = _samplePoints[0];
            a.y = heights.x;
            var b = _samplePoints[1];
            b.y = heights.y;
            var c = _samplePoints[2];
            c.y = heights.z;
            var normal = Vector3.Cross(b - a, c - a);

            if (normal.sqrMagnitude <= 1e-8f)
                return;

            normal.Normalize();
            if (Vector3.Dot(normal, transform.up) < 0f)
                normal = -normal;

            var targetRotation = Quaternion.FromToRotation(transform.up, normal) * _rigidbody.rotation;
            _rigidbody.MoveRotation(Quaternion.Slerp(_rigidbody.rotation, targetRotation,
                _rotationLerp * Time.fixedDeltaTime));
        }

        private void AlignPositionToSurface(Vector4 heights)
        {
            float averageHeight = (heights.x + heights.y + heights.z + heights.w) * 0.25f;
            float targetY = averageHeight + _pivotOffsetY;

            var position = _rigidbody.position;
            position.y = Mathf.Lerp(position.y, targetY, _positionLerp * Time.fixedDeltaTime);
            _rigidbody.position = position;
        }
    }
}
