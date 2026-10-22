# Local fixes applied to vendored sources

Upstream version records remain in versions.json. Both Linux and Android build
the same corrected files; these changes are not a new upstream version.

- efanna_graph/tests/test_nndescent.cpp: allocate a count of float elements,
  not a byte count (remove the extra sizeof(float)).
- ssg/src/index_ssg.cpp: release center and cut_graph_ arrays.
- ssg/src/util.cpp: match new[] with delete[], ensure sufficient allocation
  alignment, check allocation failure, reject invalid GenRandom size/N.

The Android-only optional KNN pool is src/knn_pooled_new_android.cpp. It is
selected by ANN_ANDROID_KNN_POOL and is never linked into normal Linux targets
or the search executable. It reduces allocation-map pressure; it is not an
algorithm or distance-scoring change.
