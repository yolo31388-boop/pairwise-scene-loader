"""场景流式加载"""
import heapq
import itertools
import threading
from dataclasses import dataclass, field


@dataclass
class SceneChunk:
    cx: int
    cy: int
    loaded: bool = False
    resources: list = field(default_factory=list)
    references: int = 0
    prewarmed: bool = False


class SceneLoader:
    # 对象池上限，超出后直接丢弃，防止无限增长
    DEFAULT_POOL_MAX = 128
    # 对象取出时需要重置的状态字段
    _RESET_FIELDS = ("x", "y", "z", "rotation", "rot_x", "rot_y", "rot_z")

    def __init__(self, max_pool_size: int = DEFAULT_POOL_MAX):
        self.chunks: dict[tuple, SceneChunk] = {}
        self.object_pool: list = []
        self.max_pool_size = max_pool_size
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._load_queue: list = []  # (priority, seq, key) 堆，数值小者优先
        self._seq = itertools.count()
        self._cancelled: set = set()
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._load_worker, name="scene-loader", daemon=True
        )
        self._worker.start()

    # ---------- bug1 修复：异步加载 + 资源预热 ----------

    def load_chunk(self, cx: int, cy: int) -> SceneChunk:
        """非阻塞：立即返回 chunk，资源在后台线程加载并预热。"""
        key = (cx, cy)
        with self._cv:
            chunk = self.chunks.get(key)
            if chunk is None:
                chunk = SceneChunk(cx, cy)
                self.chunks[key] = chunk
            self._cancelled.discard(key)
            heapq.heappush(self._load_queue, (0, next(self._seq), key))
            self._cv.notify()
        return chunk

    def _load_worker(self) -> None:
        while True:
            with self._cv:
                while not self._load_queue and not self._shutdown:
                    self._cv.wait()
                if self._shutdown:
                    return
                _, _, key = heapq.heappop(self._load_queue)
            if key in self._cancelled:
                self._cancelled.discard(key)
                continue
            self._load_and_prewarm(key)

    def _load_and_prewarm(self, key: tuple) -> None:
        # 后台线程做阻塞 IO，不卡主线程；材质/贴图在可见前预热，避免闪帧
        resources = self._fetch_resources(key)
        warmed = [self._prewarm_resource(r) for r in resources]
        with self._lock:
            chunk = self.chunks.get(key)
            if chunk is None or key in self._cancelled:
                return
            chunk.resources = warmed
            chunk.prewarmed = True
            chunk.loaded = True

    def _fetch_resources(self, key: tuple) -> list:
        # 实际项目中此处为磁盘/网络 IO
        return [{"chunk": key, "type": "geometry"}]

    def _prewarm_resource(self, resource: dict) -> dict:
        # 实际项目中此处上传材质/贴图到 GPU
        resource["prewarmed"] = True
        return resource

    # ---------- bug2 修复：卸载检查引用 + 释放内存 ----------

    def unload_chunk(self, cx: int, cy: int) -> bool:
        key = (cx, cy)
        with self._cv:
            chunk = self.chunks.get(key)
            if chunk is None:
                return True
            if chunk.references > 0:
                # 仍有引用（如玩家装备来自该区域），拒绝卸载
                return False
            # 释放资源内存，并取消未完成的加载
            chunk.resources.clear()
            del self.chunks[key]
            self._cancelled.add(key)
            return True

    # ---------- bug3 修复：取对象重置状态 + 池上限 ----------

    def get_from_pool(self) -> dict:
        if self.object_pool:
            obj = self.object_pool.pop()
            self._reset_object(obj)
            return obj
        return {"x": 0, "y": 0, "active": True}

    def return_to_pool(self, obj: dict) -> None:
        if len(self.object_pool) < self.max_pool_size:
            self.object_pool.append(obj)

    def _reset_object(self, obj: dict) -> None:
        for name in self._RESET_FIELDS:
            if name in obj:
                obj[name] = 0
        obj["active"] = True

    # ---------- bug4 修复：LOD 按屏幕占比 ----------

    def calculate_lod(self, distance: float, screen_size: float) -> int:
        # 屏幕占比 = 物体投影尺寸 / 距离，大物体即使远也占屏大
        ratio = screen_size / max(distance, 1e-6)
        if ratio >= 1.0:
            return 0
        if ratio >= 0.1:
            return 1
        return 2

    # ---------- bug5 修复：优先级调度 + 支持取消 ----------

    def schedule_load(self, chunk_key: tuple, priority: int) -> None:
        """按优先级入队（数值小者优先），由后台线程依次加载。"""
        with self._cv:
            if chunk_key not in self.chunks:
                self.chunks[chunk_key] = SceneChunk(*chunk_key)
            self._cancelled.discard(chunk_key)
            heapq.heappush(self._load_queue, (priority, next(self._seq), chunk_key))
            self._cv.notify()

    def cancel_load(self, chunk_key: tuple) -> bool:
        """取消尚未完成的加载；已加载完成的 chunk 不受影响。"""
        with self._cv:
            chunk = self.chunks.get(chunk_key)
            if chunk is not None and chunk.loaded:
                return False
            self._cancelled.add(chunk_key)
            self.chunks.pop(chunk_key, None)
            return True

    def close(self) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        self._worker.join(timeout=2)
