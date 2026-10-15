"""按仓库相对位置只读复用共享量化实现，不包含量化算法副本。"""
import sys
from pathlib import Path

# 在导入共享包前关闭字节码写入，避免在 CAGRA 之外创建 __pycache__。
sys.dont_write_bytecode = True
SHARED_ROOT = Path(__file__).resolve().parents[1] / 'ann_quantization'
if not (SHARED_ROOT / 'codecs.py').is_file():
    raise ImportError(
        f'共享量化实现不存在：{SHARED_ROOT}。'
        '请克隆完整仓库，确保 CAGRA/ 与 ann_quantization/ 位于同一父目录。'
    )
sys.path.insert(0, str(SHARED_ROOT.parent))
from ann_quantization import codecs as _shared

if Path(_shared.__file__).resolve() != (SHARED_ROOT / 'codecs.py').resolve():
    raise ImportError('共享量化实现加载路径不匹配')

# 算法和默认参数均取自共享实现；pipeline.py 仍可显式传入自己的参数。
Codec = _shared.Codec
ExactFloat = _shared.ExactFloat
SymmetricInt8 = _shared.SymmetricInt8
PCA = _shared.PCA
ProductQuantization = _shared.ProductQuantization
LocalAdaptiveInt8 = _shared.LocalAdaptiveInt8
make_codec = _shared.make_codec
save_codec = _shared.save_codec
load_codec = _shared.load_codec
