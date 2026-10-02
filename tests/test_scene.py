"""场景流式加载 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.loader import SceneLoader, SceneChunk

class TestAsyncLoad:
    def test_load_does_not_block(self):
        sl = SceneLoader()
        chunk = sl.load_chunk(0, 0)
        assert chunk is not None

class TestReferenceCheck:
    def test_cannot_unload_referenced_chunk(self):
        sl = SceneLoader()
        chunk = sl.load_chunk(0, 0)
        chunk.references = 1
        result = sl.unload_chunk(0, 0)
        assert result == False

class TestPoolReset:
    def test_pooled_object_reset(self):
        sl = SceneLoader()
        obj = {"x": 100, "y": 200, "active": False}
        sl.object_pool.append(obj)
        result = sl.get_from_pool()
        assert result["x"] == 0

class TestScreenSpaceLOD:
    def test_lod_considers_screen_size(self):
        sl = SceneLoader()
        # 远处大物体应该用高LOD
        lod = sl.calculate_lod(100, 500)
        assert lod < 2

class TestLoadPriority:
    def test_nearby_chunks_load_first(self):
        sl = SceneLoader()
        sl.schedule_load((0, 0), 1)
        sl.schedule_load((10, 10), 10)
        assert (0, 0) in sl.chunks
