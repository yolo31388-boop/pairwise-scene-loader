"""场景流式加载。

修复点：
1. load_chunk 异步加载（后台线程），加载完成后自动预热材质/贴图资源，
   不再卡住主线程；需要确定性时用 wait_for_loads() / process_loaded()。
2. unload_chunk 先检查引用计数，被引用的块拒绝卸载；卸载时释放资源
   占用的（GPU/内存）资源并回收内存。
3. 对象池取对象时重置全部状态（位置/旋转/激活标记），且池子有上限，
   超上限归还的对象直接销毁。
4. LOD 按屏幕占比计算（屏幕上越大模型越精细），距离只作为小屏占时的
   辅助判断。
5. schedule_load 按优先级（数值越小越优先）调度，可通过 cancel_load
   取消尚未开始/正在进行的加载。
"""
from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

_MISSING = object()


@dataclass
class Resource:
    """一个美术资源（材质/贴图/模型）。"""

    name: str
    size: int = 0
    loaded: bool = False
    preheated: bool = False


@dataclass
class SceneChunk:
    cx: int
    cy: int
    loaded: bool = False
    preheated: bool = False
    resources: list = field(default_factory=list)
    references: int = 0


class SceneLoader:
    def __init__(
        self,
        chunk_loader: Optional[Callable[[int, int], list]] = None,
        pool_max_size: int = 64,
        async_loader: bool = True,
        max_pool_size: Optional[int] = None,
    ):
        self.chunks: dict[tuple, SceneChunk] = {}
        self.object_pool: list = []
        self.pool_max_size = (
            max_pool_size if max_pool_size is not None else pool_max_size
        )
        self.max_pool_size = self.pool_max_size

        # 加载任务队列：(priority, seq, chunk_key)
        self._queue: list = []
        self._seq = 0
        self._pending: set = set()
        self._inflight: set = set()
        self._cancelled: set = set()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._chunk_loader = chunk_loader
        self._async = async_loader
        self._worker: Optional[threading.Thread] = None
        self._closed = False

        # 加载完成顺序（便于测试断言优先级是否生效）
        self.load_order: list = []
        # 被释放资源的记录（便于测试断言内存是否回收）
        self.released_resources: list = []
        # 已取消加载的块
        self.cancelled: set = set()
        # 当前被资源占用的内存（字节级别的模拟值）
        self.memory_usage: int = 0

    # ------------------------------------------------------------------
    # 1. 异步加载 + 资源预热
    # ------------------------------------------------------------------
    def load_chunk(
        self,
        cx: int,
        cy: int,
        priority: int = 0,
        blocking: Optional[bool] = None,
    ) -> SceneChunk:
        """请求加载一个场景块。

        默认异步：返回时块可能尚未 loaded，加载在后台线程进行，完成后会
        自动预热资源。传 blocking=True 可同步等待本次加载完成。
        """
        key = (cx, cy)
        with self._lock:
            existing = self.chunks.get(key)
            if existing is not None and existing.loaded:
                return existing
            if existing is None:
                chunk = SceneChunk(cx, cy)
                self.chunks[key] = chunk
            else:
                chunk = existing
                self._cancelled.discard(key)
            if key not in self._pending and key not in self._inflight:
                self._enqueue(key, priority)
            if blocking is None:
                blocking = not self._async

        if blocking:
            self._ensure_worker()
            self.wait_for_loads(key)
        else:
            self._ensure_worker()
        return chunk

    # 语义别名（需求描述里的函数名）
    load_scene_chunk = load_chunk

    def schedule_load(self, chunk_key: tuple, priority: int = 0) -> tuple:
        """按优先级调度一次加载（数值越小越优先）。返回任务句柄。"""
        with self._lock:
            if chunk_key not in self.chunks:
                self.chunks[chunk_key] = SceneChunk(*chunk_key)
            self._cancelled.discard(chunk_key)
            self._enqueue(chunk_key, priority)
            handle = chunk_key
        self._ensure_worker()
        return handle

    def cancel_load(self, chunk_key: tuple) -> bool:
        """取消尚未完成的加载；已经加载完成的块不会被回滚。"""
        with self._lock:
            chunk = self.chunks.get(chunk_key)
            if chunk is not None and chunk.loaded:
                return False
            self._cancelled.add(chunk_key)
            self._pending.discard(chunk_key)
            self.cancelled.add(chunk_key)
            cancelled = True
        if cancelled:
            # 唤醒worker：若该任务正在执行，它会在完成前检查取消标记
            with self._cond:
                self._cond.notify_all()
        return cancelled

    def wait_for_loads(self, chunk_key: Optional[tuple] = None, timeout: Optional[float] = None) -> bool:
        """等待（指定或全部）排队中的加载完成。返回是否全部完成。"""
        if not self._async:
            self.process_loaded()
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._pending or self._inflight:
                if chunk_key is not None and chunk_key not in self._pending and chunk_key not in self._inflight:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    # 同步模式下手动推进加载队列
    def process_loaded(self) -> int:
        processed = 0
        while True:
            with self._lock:
                if not self._queue:
                    return processed
                _, _, key = heapq.heappop(self._queue)
                self._pending.discard(key)
            self._run_job(key)
            processed += 1

    update = process_loaded

    @property
    def is_idle(self) -> bool:
        with self._lock:
            return not self._pending and not self._inflight

    def _enqueue(self, key: tuple, priority: int) -> None:
        self._seq += 1
        heapq.heappush(self._queue, (priority, self._seq, key))
        self._pending.add(key)
        self._cancelled.discard(key)

    def _ensure_worker(self) -> None:
        if not self._async:
            return
        with self._lock:
            if self._closed:
                return
            if self._worker is None or not self._worker.is_alive():
                worker = threading.Thread(
                    target=self._worker_loop, name="scene-loader", daemon=True
                )
                self._worker = worker
                worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait(timeout=0.1)
                if self._closed and not self._queue:
                    return
                _, _, key = heapq.heappop(self._queue)
                self._pending.discard(key)
                if key in self._cancelled or key not in self.chunks:
                    self._cancelled.discard(key)
                    self._cond.notify_all()
                    continue
                self._inflight.add(key)
            try:
                self._run_job(key)
            finally:
                with self._cond:
                    self._inflight.discard(key)
                    self._cond.notify_all()

    def _run_job(self, key: tuple) -> None:
        cx, cy = key
        resources = self._fetch_resources(cx, cy)
        with self._lock:
            # 加载过程中被取消/卸载：丢弃结果并释放已取出的资源
            if key in self._cancelled or key not in self.chunks:
                self._cancelled.discard(key)
                for res in resources:
                    self._release_resource(res)
                self.cancelled.add(key)
                return
            chunk = self.chunks[key]
            # 可能已被其他路径同步加载
            if chunk.loaded:
                return
            chunk.resources = resources
            chunk.loaded = True
            self.load_order.append(key)
            for res in resources:
                res.loaded = True
                self.memory_usage += getattr(res, "size", 0)
        # 预热放在锁外（GPU 上传是耗时操作，不能挡后续任务的调度）
        self.preheat_resources(cx, cy)

    def _fetch_resources(self, cx: int, cy: int) -> list:
        """获取块的资源列表；测试可通过构造函数注入自定义加载器。"""
        if self._chunk_loader is not None:
            resources = self._chunk_loader(cx, cy)
            return list(resources)
        return [Resource(f"chunk_{cx}_{cy}_asset")]

    def preheat_resources(self, cx: int, cy: int) -> bool:
        """把块内材质/贴图提前上传/编译，避免首帧看到物体时才加载。"""
        with self._lock:
            chunk = self.chunks.get((cx, cy))
            if chunk is None or not chunk.loaded or chunk.preheated:
                return False
            for res in chunk.resources:
                if not getattr(res, "preheated", False):
                    res.preheated = True
            chunk.preheated = True
            return True

    # ------------------------------------------------------------------
    # 2. 卸载：引用检查 + 内存释放
    # ------------------------------------------------------------------
    def unload_chunk(self, cx: int, cy: int, force: bool = False) -> bool:
        """卸载场景块。

        仍被引用（references > 0，例如玩家身上的装备来自该块）时拒绝
        卸载并返回 False；正常卸载时释放全部资源占用的内存。
        """
        key = (cx, cy)
        with self._lock:
            chunk = self.chunks.get(key)
            if chunk is None:
                return False
            if chunk.references > 0 and not force:
                return False
            # 仍在排队/加载中：取消后不再卸载（块尚未可见）
            self._cancelled.add(key)
            resources = chunk.resources
            del self.chunks[key]
            self._pending.discard(key)
        for res in resources:
            self._release_resource(res)
        return True

    unload_scene_chunk = unload_chunk

    def unload_distant(self, center: tuple, radius: float) -> list:
        """卸载以 center 为中心、radius 之外且无引用的块，返回卸载的键。"""
        unloaded = []
        with self._lock:
            keys = [
                key
                for key, chunk in self.chunks.items()
                if chunk.references <= 0
                and abs(key[0] - center[0]) ** 2 + abs(key[1] - center[1]) ** 2
                > radius * radius
            ]
        for key in keys:
            if self.unload_chunk(key[0], key[1]):
                unloaded.append(key)
        return unloaded

    def _release_resource(self, res) -> None:
        if getattr(res, "loaded", False):
            self.memory_usage -= getattr(res, "size", 0)
        res.loaded = False
        res.preheated = False
        self.released_resources.append(getattr(res, "name", res))

    # 引用计数辅助（装备/技能引用某块的资源时加引用）
    def add_reference(self, cx: int, cy: int, count: int = 1) -> None:
        chunk = self.chunks.get((cx, cy))
        if chunk is not None:
            chunk.references += count

    def release_reference(self, cx: int, cy: int, count: int = 1) -> None:
        chunk = self.chunks.get((cx, cy))
        if chunk is not None:
            chunk.references = max(0, chunk.references - count)

    # ------------------------------------------------------------------
    # 3. 对象池：取用时重置状态 + 池上限
    # ------------------------------------------------------------------
    def get_from_pool(self, **overrides) -> dict:
        """从池中取对象，位置/旋转/激活状态一律重置，杜绝串状态。"""
        with self._lock:
            obj = self.object_pool.pop() if self.object_pool else self._new_object()
        self.reset_object(obj)
        obj.update(overrides)
        return obj

    @staticmethod
    def _new_object() -> dict:
        return {"x": 0.0, "y": 0.0, "active": True}

    @staticmethod
    def reset_object(obj: dict) -> dict:
        obj["x"] = 0.0
        obj["y"] = 0.0
        obj["active"] = True
        # 旋转字段可能叫 rotation / rot / angle
        for name in ("rotation", "rot", "angle"):
            if name in obj:
                obj[name] = 0.0
        return obj

    def return_to_pool(self, obj: dict) -> bool:
        """归还对象；池子已满（上限）时直接销毁，返回 False。"""
        with self._lock:
            if len(self.object_pool) >= self.pool_max_size:
                return False
            self.object_pool.append(obj)
            return True

    release_to_pool = return_to_pool

    def pool_size(self) -> int:
        with self._lock:
            return len(self.object_pool)

    # ------------------------------------------------------------------
    # 4. LOD：以屏幕占比为主
    # ------------------------------------------------------------------
    def calculate_lod(self, distance: float, screen_size: float) -> int:
        """按屏幕占比选择 LOD（0=高模，1=中模，2=低模）。

        屏幕占比足够大时，无论多远都用高模（远处的大山不该变马赛克）；
        屏幕占比小但距离近时，用中模而不是直接低模。
        """
        if screen_size >= 200:
            return 0
        if screen_size >= 50:
            return 1
        if distance < 10:
            return 1
        return 2

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
