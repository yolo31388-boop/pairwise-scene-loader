"""场景加载表驱动测试：失败回滚、引用计数、缓存、切换、零引用释放。"""

from __future__ import annotations

import pytest

from src import SceneLoader, Scene


def _eng():
    eng = SceneLoader()
    eng.add_scene(Scene("s1", ["model_a", "tex_b", "sfx_c"]))
    eng.add_scene(Scene("s2", ["fail_x", "model_a"]))
    return eng


def test_basic_load():
    eng = _eng()
    assert eng.load_scene("s1") is True
    assert eng.loaded_count() == 3


def test_load_failure_rollback():
    """加载失败必须回滚已加载资源。"""
    eng = _eng()
    result = eng.load_scene("s2")  # fail_x会失败
    assert result is False
    # model_a在fail_x之前加载了，应该被回滚
    r = eng.get_resource("model_a")
    assert r is None or r.ref_count == 0 or not r.loaded, \
        "加载失败应回滚已加载资源: ref=%d loaded=%s" % (r.ref_count if r else -1, r.loaded if r else "none")
    assert eng.loaded_count() == 0, "回滚后不应有加载的资源"


def test_resource_caching():
    eng = _eng()
    eng.load_scene("s1")
    r1 = eng.get_resource("model_a")
    ref_before = r1.ref_count
    # 加载另一个共享model_a的场景
    eng.add_scene(Scene("s3", ["model_a", "other"]))
    eng.load_scene("s3")
    assert eng.get_resource("model_a").ref_count == ref_before + 1


def test_switch_scene():
    eng = _eng()
    eng.load_scene("s1")
    eng.add_scene(Scene("s3", ["model_d"]))
    eng.switch_scene("s3")
    assert eng.current_scene == "s3"
    # s1的资源应该被卸载（ref减1）


def test_zero_ref_releases():
    eng = _eng()
    eng.load_scene("s1")
    eng.unload_scene("s1")
    for rid in ["model_a", "tex_b", "sfx_c"]:
        r = eng.get_resource(rid)
        assert r is None or not r.loaded or r.ref_count <= 0


def test_unload_nonexistent():
    eng = _eng()
    eng.unload_scene("nonexistent")  # 不崩溃


def test_load_nonexistent_scene():
    eng = _eng()
    assert eng.load_scene("nope") is False


def test_ref_count_precise():
    eng = _eng()
    eng.load_scene("s1")
    r = eng.get_resource("model_a")
    assert r.ref_count == 1
    eng.unload_scene("s1")
    assert r.ref_count == 0


def test_cancel_load_rollback():
    """取消加载（模拟失败）应回滚。"""
    eng = _eng()
    eng.add_scene(Scene("s4", ["r1", "r2", "fail_z"]))
    eng.load_scene("s4")
    assert eng.loaded_count() == 0


def test_deterministic():
    def run():
        eng = _eng()
        eng.load_scene("s1")
        eng.unload_scene("s1")
        return eng.snapshot()
    assert run() == run()
