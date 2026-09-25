"""场景异步加载引擎：事务加载、失败回滚、引用计数、资源缓存、场景切换。

核心不变量：
- 加载事务性，失败回滚已加载资源
- 取消加载回滚
- 引用计数精确（加载+1，回滚/卸载-1，不泄漏不重复释放）
- 资源缓存（重复加载命中缓存，不重复加载）
- 场景切换先卸载旧场景
- 引用计数为 0 立即释放
- 资源依赖自动加载（A 依赖 B，加载 A 时自动加载 B）
- 加载进度准确反映已加载/总资源
- 加载失败返回明确错误信息
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple


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


class SceneLoader:
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.scenes: Dict[str, Scene] = {}
        self.current_scene: Optional[str] = None
        self.load_errors: List[str] = []
        self.last_error: Optional[str] = None
        # 资源依赖：rid -> 依赖的 rid 列表
        self.dependencies: Dict[str, List[str]] = {}
        # 实际从源加载的次数（缓存命中不增加），用于验证缓存
        self.load_counts: Dict[str, int] = {}
        # 加载进度
        self.progress_loaded: int = 0
        self.progress_total: int = 0
        # 每个资源加载完成后的钩子（模拟异步加载中的取消点）
        self.on_resource_loaded: Optional[Callable[[str], None]] = None
        self._cancel_requested: bool = False

    def add_scene(self, s: Scene) -> None:
        self.scenes[s.sid] = s

    def set_dependencies(self, rid: str, deps: List[str]) -> None:
        self.dependencies[rid] = list(deps)

    def cancel_load(self) -> None:
        """请求取消当前加载，已加载部分将被回滚。"""
        self._cancel_requested = True

    def load_progress(self) -> Tuple[int, int]:
        """返回 (已加载资源数, 总资源数)。"""
        return self.progress_loaded, self.progress_total

    def _expand_resources(self, rids: List[str]) -> List[str]:
        """展开资源列表（含传递依赖），依赖在前，去重且保持确定性顺序。"""
        order: List[str] = []
        seen: Set[str] = set()

        def visit(rid: str) -> None:
            if rid in seen:
                return
            seen.add(rid)
            for dep in self.dependencies.get(rid, []):
                visit(dep)
            order.append(rid)

        for rid in rids:
            visit(rid)
        return order

    def _load_resource(self, rid: str) -> bool:
        # 缓存命中：已加载资源直接引用计数 +1，不重复加载
        if rid in self.resources and self.resources[rid].loaded:
            self.resources[rid].ref_count += 1
            return True
        if rid.startswith("fail_"):
            return False
        r = self.resources.get(rid)
        if r is None:
            r = Resource(rid, "generic")
            self.resources[rid] = r
        r.loaded = True
        r.ref_count += 1
        self.load_counts[rid] = self.load_counts.get(rid, 0) + 1
        return True

    def _release_resource(self, rid: str) -> None:
        """引用计数 -1；为 0 时立即释放。重复释放是安全的空操作。"""
        r = self.resources.get(rid)
        if r is None or r.ref_count <= 0:
            return
        r.ref_count -= 1
        if r.ref_count == 0:
            r.loaded = False
            del self.resources[rid]

    def rollback_load(self, loaded: List[str]) -> None:
        """回滚一次加载事务：按相反顺序释放已加载资源并重置进度。"""
        for rid in reversed(loaded):
            self._release_resource(rid)
        self.progress_loaded = 0

    def load_scene(self, sid: str) -> bool:
        """事务性加载场景：全部依赖资源成功才提交，失败/取消回滚。"""
        scene = self.scenes.get(sid)
        if scene is None:
            self.last_error = f"scene not found: {sid}"
            self.load_errors.append(self.last_error)
            return False
        if scene.loaded:
            # 缓存：同一场景重复加载不重复加载资源
            return True
        plan = self._expand_resources(scene.resources)
        self.progress_total = len(plan)
        self.progress_loaded = 0
        self._cancel_requested = False
        loaded_so_far: List[str] = []
        for rid in plan:
            if self._cancel_requested:
                self.last_error = f"load cancelled for scene: {sid}"
                self.load_errors.append(self.last_error)
                self.rollback_load(loaded_so_far)
                return False
            if self._load_resource(rid):
                loaded_so_far.append(rid)
                self.progress_loaded += 1
                if self.on_resource_loaded is not None:
                    self.on_resource_loaded(rid)
            else:
                self.last_error = f"failed to load resource: {rid} (scene: {sid})"
                self.load_errors.append(self.last_error)
                self.rollback_load(loaded_so_far)
                return False
        scene.loaded = True
        self.current_scene = sid
        return True

    def unload_scene(self, sid: str) -> None:
        scene = self.scenes.get(sid)
        if scene is None or not scene.loaded:
            return
        for rid in self._expand_resources(scene.resources):
            self._release_resource(rid)
        scene.loaded = False
        if self.current_scene == sid:
            self.current_scene = None

    def switch_scene(self, sid: str) -> bool:
        """先卸载旧场景，再加载新场景。"""
        if self.current_scene:
            self.unload_scene(self.current_scene)
        return self.load_scene(sid)

    def get_resource(self, rid: str) -> Optional[Resource]:
        return self.resources.get(rid)

    def loaded_count(self) -> int:
        return sum(1 for r in self.resources.values() if r.loaded)

    def snapshot(self) -> dict:
        return {
            "current": self.current_scene,
            "resources": {rid: {"loaded": r.loaded, "ref": r.ref_count}
                          for rid, r in self.resources.items()},
            "scenes": {sid: s.loaded for sid, s in self.scenes.items()},
        }
