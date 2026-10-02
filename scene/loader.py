"""场景流式加载 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class SceneChunk:
    cx: int
    cy: int
    loaded: bool = False
    resources: list = field(default_factory=list)
    references: int = 0

class SceneLoader:
    def __init__(self):
        self.chunks: dict[tuple, SceneChunk] = {}
        self.object_pool: list = []  # bug3: 不重置状态

    def load_chunk(self, cx: int, cy: int) -> SceneChunk:
        # bug1: 同步加载
        chunk = SceneChunk(cx, cy, loaded=True)
        self.chunks[(cx, cy)] = chunk
        return chunk

    def unload_chunk(self, cx: int, cy: int) -> bool:
        # bug2: 不检查引用，不释放内存
        key = (cx, cy)
        if key in self.chunks:
            del self.chunks[key]
        return True

    def get_from_pool(self) -> dict:
        # bug3: 不重置状态
        if self.object_pool:
            return self.object_pool.pop()
        return {"x": 0, "y": 0, "active": True}

    def calculate_lod(self, distance: float, screen_size: float) -> int:
        # bug4: 只按距离
        if distance < 10:
            return 0
        elif distance < 50:
            return 1
        return 2

    def schedule_load(self, chunk_key: tuple, priority: int) -> None:
        # bug5: 无优先级，不支持取消
        self.load_chunk(chunk_key[0], chunk_key[1])
