"""Prepare a memory-only DiskANN profile without editing vendor sources."""
from pathlib import Path
import shutil
import sys
source, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
for original in (source / 'include').rglob('*'):
    if original.is_file() and original.name != 'aligned_file_reader.h':
        target = output / 'include' / original.relative_to(source / 'include')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != original.read_bytes():
            shutil.copy2(original, target)
def once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('Upstream source changed: ' + before[:100])
    return text.replace(before, after, 1)
p = output / 'include/aligned_file_reader.h'
text = once((source / 'include/aligned_file_reader.h').read_text(), '#include <libaio.h>', '#include <cstdint>')
text = once(text, 'typedef io_context_t IOContext;', 'using IOContext = uintptr_t; // opaque; no AIO backend in memory profile')
put(p, text)
text = (source / 'src/index_factory.cpp').read_text()
start = text.index('{', text.index('IndexFactory::construct_pq_datastore('))
depth, end = 1, start + 1
while depth:
    depth += (text[end] == '{') - (text[end] == '}')
    end += 1
text = text[:start] + '''{
    throw ANNException("Memory-only profile supports --build_PQ_bytes 0 only. "
                       "PQ/OPQ graph construction is not compiled in. "
                       "The separate vamana_search quantized scorer is unaffected.", -1);
}''' + text[end:]
text = once(text, '    check_config();', '''    check_config();
    if (_config->pq_dist_build || _config->use_opq)
        throw ANNException("Memory-only profile: PQ/OPQ graph construction is unsupported", -1);''')
for kind in ['float', 'int8_t', 'uint8_t']:
    text += ('\nnamespace diskann { template std::shared_ptr<PQDataStore<' + kind + '>> '
             'IndexFactory::construct_pq_datastore<' + kind + '>(DataStoreStrategy, size_t, size_t, Metric, size_t, bool); }\n')
put(output / 'index_factory_memory.cpp', text)
text = (source / 'src/distance.cpp').read_text()
assert text.count('const float *last = a + size;') == 2
text = text.replace('const float *last = a + size;', 'const T *last = a + size;')
assert text.count('const float *unroll_group = last - 3;') == 2
text = text.replace('const float *unroll_group = last - 3;', 'const T *unroll_group = last - 3;')
put(output / 'distance_portable.cpp', text)
# Index still owns an unused legacy PQ table. Keep its real upstream lifetime
# methods verbatim, without importing unrelated PQ training / BLAS functions.
pq = (source / 'src/pq.cpp').read_text()
start = pq.index('FixedChunkPQTable::FixedChunkPQTable()')
end = pq.index('#ifdef EXEC_ENV_OLS\nvoid FixedChunkPQTable::load_pq_centroid_bin', start)
put(output / 'pq_table_lifetime.cpp', '#include "pq.h"\nnamespace diskann {\n' + pq[start:end] + '\n}\n')
