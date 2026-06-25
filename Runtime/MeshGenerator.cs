using UnityEngine;
using UnityEngine.Rendering;

namespace YellowSquad.OceanLib
{
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class MeshGenerator : MonoBehaviour
    {
        [SerializeField, Range(1, 100000)] private int _planeSize = 1000;
        [SerializeField, Range(1, 1000)] private int _cellSize = 10;

        private void Awake()
        {
            int cellsPerRow = Mathf.Max(1, _planeSize / _cellSize);

            var mesh = new Mesh
            {
                name = "Ocean",
                indexFormat = IndexFormat.UInt32,
                vertices = GenerateVertices(cellsPerRow),
                triangles = GenerateTriangles(cellsPerRow),
            };
            mesh.RecalculateBounds();

            GetComponent<MeshFilter>().mesh = mesh;
        }

        private Vector3[] GenerateVertices(int cellsPerRow)
        {
            int verticesPerRow = cellsPerRow + 1;
            float halfLength = _planeSize * 0.5f;
            float spacing = _planeSize / (float)cellsPerRow;

            var vertices = new Vector3[verticesPerRow * verticesPerRow];

            for (int i = 0, z = 0; z < verticesPerRow; z++)
            {
                for (int x = 0; x < verticesPerRow; x++, i++)
                {
                    vertices[i] = new Vector3(x * spacing - halfLength, 0f, z * spacing - halfLength);
                }
            }

            return vertices;
        }

        private static int[] GenerateTriangles(int cellsPerRow)
        {
            int verticesPerRow = cellsPerRow + 1;
            int[] triangles = new int[cellsPerRow * cellsPerRow * 6];

            for (int ti = 0, vi = 0, z = 0; z < cellsPerRow; z++, vi++)
            {
                for (int x = 0; x < cellsPerRow; x++, ti += 6, vi++)
                {
                    triangles[ti] = vi;
                    triangles[ti + 3] = triangles[ti + 2] = vi + 1;
                    triangles[ti + 4] = triangles[ti + 1] = vi + verticesPerRow;
                    triangles[ti + 5] = vi + verticesPerRow + 1;
                }
            }

            return triangles;
        }
    }
}
