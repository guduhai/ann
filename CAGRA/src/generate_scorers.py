"""Generate the same 96-D shared-header specialization used by the DEEP runners."""
import hashlib
import json
from pathlib import Path
import re
import sys
source, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
manifest = {}
for name in ('native_scorer.hpp', 'native_variable_scorer.hpp', 'native_search_scorer.hpp'):
    original = (source / name).read_bytes()
    body = (source / name).read_text(encoding='utf-8')
    if name != 'native_search_scorer.hpp':
        body = re.sub(r'\b128\b', '96', body)
    if name == 'native_scorer.hpp':
        body = body.replace('stride_ = 129', 'stride_ = 97')
        body = body.replace('16 * 256 * 8 * 4', '16 * 256 * 6 * 4').replace('d < 8;', 'd < 6;')
        body = body.replace('256 + center) * 8', '256 + center) * 6').replace('query[sub * 8', 'query[sub * 6')
    (output / name).write_text(body, encoding='utf-8')
    manifest[name] = {'source_sha256': hashlib.sha256(original).hexdigest(),
                      'adapted_sha256': hashlib.sha256(body.encode()).hexdigest()}
(output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
