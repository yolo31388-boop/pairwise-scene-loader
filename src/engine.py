"""场景异步加载引擎：事务加载、失败/取消回滚、引用计数、资源缓存、依赖、进度。

核心不变量：
- 场景加载是事务性的：全部资源（含传递依赖）加载成功才提交
- 任一资源失败 / 加载被取消：回滚本事务已获取的全部资源并减少引用计数
- 引用计数精确：获取 +1，回滚/卸载 -1；归零立即释放；不会重复释放
- 已加载资源缓存命中时不重复加载（仅 +1）
- 场景切换先卸载旧场景（释放其全部持有资源）再加载新场景
- 资源依赖自动处理：加载 A 时先加载 A 依赖的 B（支持传递依赖、去重、环保护）
- 进度 = 已获取资源数 / 事务所需资源总数（含依赖）
- 失败记录明确错误信息（last_error / load_errors），不静默失败
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class Resource:
    rid: str
    rtype: str
    ref_count: int = 0
    loaded: bool = False


@dataclass
class Scene:
    sid: str
    resources: List[str] = field(default_factory=list)
    loaded: bool = False


@dataclass
class CancelToken:
    """异步加载取消令牌：在任意资源边界置位即可取消本次加载。"""

    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


ProgressCallback = Callable[[int, int], None]


class SceneLoader:
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.scenes: Dict[str, Scene] = {}
        # 资源依赖表：rid -> 它所依赖的资源 id 列表
        self.dependencies: Dict[str, List[str]] = {}
        self.current_scene: Optional[str] = None
        self.load_errors: List[str] = []
        self.last_error: Optional[str] = None
        # 实际从“磁盘”加载的次数（缓存命中不计），用于验证不重复加载
        self._fetch_count: Dict[str, int] = {}
        # 每个已提交场景实际持有的资源（提交时的依赖闭包，按加载顺序）
        self._held: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    # 注册
    # ------------------------------------------------------------------ #
    def add_scene(self, s: Scene) -> None:
        self.scenes[s.sid] = s

    def add_dependency(self, rid: str, depends_on: str) -> None:
        """声明资源 rid 依赖 depends_on；加载 rid 时会先自动加载 depends_on。"""
        self.dependencies.setdefault(rid, []).append(depends_on)

    # ------------------------------------------------------------------ #
    # 依赖闭包
    # ------------------------------------------------------------------ #
    def _resolve_closure(self, roots: Iterable[str]) -> List[str]:
        """返回去重后的资源列表，依赖在前、依赖者在后（深度优先后序）。"""
        ordered: List[str] = []
        seen: set[str] = set()

        def visit(rid: str, stack: set[str]) -> None:
            if rid in seen or rid in stack:  # stack 命中即依赖环，剪枝防死循环
                return
            stack.add(rid)
            for dep in self.dependencies.get(rid, []):
                visit(dep, stack)
            stack.discard(rid)
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)

        for root in roots:
            visit(root, set())
        return ordered

    def required_resources(self, sid: str) -> List[str]:
        """场景提交所需的全部资源（含传递依赖），加载顺序。"""
        scene = self.scenes.get(sid)
        if scene is None:
            return []
        return self._resolve_closure(scene.resources)

    # ------------------------------------------------------------------ #
    # 资源获取 / 释放（引用计数）
    # ------------------------------------------------------------------ #
    def _load_resource(self, rid: str) -> bool:
        """获取资源：缓存命中只 +1；未加载则真正加载并 +1。失败返回 False。"""
        r = self.resources.get(rid)
        if r is not None and r.loaded:
            # 缓存命中：不重复加载，仅增加引用
            r.ref_count += 1
            return True
        if rid.startswith("fail_"):
            return False
        if r is None:
            r = Resource(rid, "generic")
            self.resources[rid] = r
        r.loaded = True
        r.ref_count += 1
        self._fetch_count[rid] = self._fetch_count.get(rid, 0) + 1
        return True

    def _release_resource(self, rid: str) -> bool:
        """释放一次引用：-1；引用归零立即释放资源。重复释放安全（返回 False）。"""
        r = self.resources.get(rid)
        if r is None or not r.loaded or r.ref_count <= 0:
            return False
        r.ref_count -= 1
        if r.ref_count <= 0:
            r.ref_count = 0
            r.loaded = False  # 引用为 0，立即释放
        return True

    def rollback_load(self, acquired: Iterable[str]) -> int:
        """回滚一次未提交的加载：释放本事务已获取的全部资源并减少引用计数。

        每个资源在事务内只获取一次，故只释放一次；返回实际释放的资源数。
        """
        released = 0
        for rid in acquired:
            if self._release_resource(rid):
                released += 1
        return released

    # ------------------------------------------------------------------ #
    # 事务加载（同步入口）
    # ------------------------------------------------------------------ #
    def load_scene(
        self,
        sid: str,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> bool:
        """事务性加载场景：全部资源成功才提交，否则整体回滚。"""
        scene = self.scenes.get(sid)
        if scene is None:
            self._record_error(f"scene '{sid}' not found")
            return False

        if scene.loaded:
            # 场景已提交：重复加载不重复获取资源
            held = self._held.get(sid, scene.resources)
            if on_progress is not None:
                on_progress(len(held), len(held))
            return True

        required = self._resolve_closure(scene.resources)
        total = len(required)
        acquired: List[str] = []

        for index, rid in enumerate(required, start=1):
            if not self._load_resource(rid):
                self.rollback_load(acquired)
                self._record_error(
                    f"failed to load resource '{rid}' required by scene '{sid}'; "
                    f"rolled back {len(acquired)} already-loaded resource(s)"
                )
                if on_progress is not None:
                    on_progress(len(acquired), total)
                return False
            acquired.append(rid)
            if on_progress is not None:
                on_progress(index, total)

        # 全部成功，提交事务
        scene.loaded = True
        self._held[sid] = required
        self.current_scene = sid
        self.last_error = None
        return True

    # ------------------------------------------------------------------ #
    # 异步加载（支持取消；标准库 asyncio）
    # ------------------------------------------------------------------ #
    async def load_scene_async(
        self,
        sid: str,
        *,
        cancel_token: Optional[CancelToken] = None,
        on_progress: Optional[ProgressCallback] = None,
        step_delay: float = 0.0,
    ) -> bool:
        """事务性异步加载。每个资源边界检查取消；取消或失败均整体回滚。"""
        scene = self.scenes.get(sid)
        if scene is None:
            self._record_error(f"scene '{sid}' not found")
            return False

        if scene.loaded:
            held = self._held.get(sid, scene.resources)
            if on_progress is not None:
                on_progress(len(held), len(held))
            return True

        required = self._resolve_closure(scene.resources)
        total = len(required)
        acquired: List[str] = []

        for index, rid in enumerate(required, start=1):
            if cancel_token is not None and cancel_token.cancelled:
                return self._abort_load(sid, acquired, total, on_progress, cancelled=True)
            if step_delay:
                await asyncio.sleep(step_delay)
            if cancel_token is not None and cancel_token.cancelled:
                return self._abort_load(sid, acquired, total, on_progress, cancelled=True)
            if not self._load_resource(rid):
                self.rollback_load(acquired)
                self._record_error(
                    f"failed to load resource '{rid}' required by scene '{sid}'; "
                    f"rolled back {len(acquired)} already-loaded resource(s)"
                )
                if on_progress is not None:
                    on_progress(len(acquired), total)
                return False
            acquired.append(rid)
            if on_progress is not None:
                on_progress(index, total)

        scene.loaded = True
        self._held[sid] = required
        self.current_scene = sid
        self.last_error = None
        return True

    def _abort_load(
        self,
        sid: str,
        acquired: List[str],
        total: int,
        on_progress: Optional[ProgressCallback],
        *,
        cancelled: bool,
    ) -> bool:
        released = self.rollback_load(acquired)
        reason = "cancelled" if cancelled else "aborted"
        self._record_error(
            f"scene '{sid}' loading {reason}; rolled back {released}/{total} resource(s)"
        )
        if on_progress is not None:
            on_progress(len(acquired), total)
        return False

    # ------------------------------------------------------------------ #
    # 卸载与场景切换
    # ------------------------------------------------------------------ #
    def unload_scene(self, sid: str) -> None:
        """卸载场景：释放提交时持有的全部资源（含依赖），各 -1。"""
        scene = self.scenes.get(sid)
        if scene is None or not scene.loaded:
            return
        held = self._held.pop(sid, scene.resources)
        for rid in held:
            self._release_resource(rid)
        scene.loaded = False
        if self.current_scene == sid:
            self.current_scene = None

    def switch_scene(self, sid: str) -> bool:
        """先卸载旧场景（减少其全部资源引用），再事务性加载新场景。"""
        if self.current_scene is not None:
            self.unload_scene(self.current_scene)
        return self.load_scene(sid)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get_resource(self, rid: str) -> Optional[Resource]:
        return self.resources.get(rid)

    def fetch_count(self, rid: str) -> int:
        """资源实际加载次数（缓存命中不计）。"""
        return self._fetch_count.get(rid, 0)

    def loaded_count(self) -> int:
        return sum(1 for r in self.resources.values() if r.loaded)

    def _record_error(self, message: str) -> None:
        self.last_error = message
        self.load_errors.append(message)

    def snapshot(self) -> dict:
        return {
            "current": self.current_scene,
            "resources": {
                rid: {"loaded": r.loaded, "ref": r.ref_count}
                for rid, r in self.resources.items()
            },
            "scenes": {sid: s.loaded for sid, s in self.scenes.items()},
        }
