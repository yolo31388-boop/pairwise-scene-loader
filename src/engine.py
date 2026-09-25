"""场景异步加载引擎：事务加载、失败回滚、引用计数、资源缓存、场景切换。

核心不变量：
- 加载事务性，失败回滚已加载资源
- 取消加载回滚
- 引用计数精确
- 资源缓存
- 场景切换卸载旧场景
- 引用零释放
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


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

    def add_scene(self, s: Scene) -> None:
        self.scenes[s.sid] = s

    def _load_resource(self, rid: str) -> bool:
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
        return True

    def load_scene(self, sid: str) -> bool:
        """加载场景。BUG：失败不回滚已加载资源。"""
        scene = self.scenes.get(sid)
        if scene is None:
            return False
        loaded_so_far: List[str] = []
        for rid in scene.resources:
            if self._load_resource(rid):
                loaded_so_far.append(rid)
            else:
                self.load_errors.append(f"failed to load {rid}")
                # BUG：不回滚已加载的资源
                # 正确：for r in loaded_so_far: self._unload_resource(r)
                return False
        scene.loaded = True
        self.current_scene = sid
        return True

    def unload_scene(self, sid: str) -> None:
        scene = self.scenes.get(sid)
        if scene is None:
            return
        for rid in scene.resources:
            r = self.resources.get(rid)
            if r:
                r.ref_count -= 1
                if r.ref_count <= 0:
                    r.loaded = False
        scene.loaded = False
        if self.current_scene == sid:
            self.current_scene = None

    def switch_scene(self, sid: str) -> bool:
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
