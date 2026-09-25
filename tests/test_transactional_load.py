"""事务性加载补充测试：回滚、取消、引用计数、缓存、依赖、进度、错误信息。"""

from __future__ import annotations

from src import SceneLoader, Scene


def _eng():
    eng = SceneLoader()
    eng.add_scene(Scene("s1", ["model_a", "tex_b", "sfx_c"]))
    eng.add_scene(Scene("s2", ["model_a", "fail_tex", "sfx_c"]))
    return eng


def test_normal_load_commits_all():
    eng = _eng()
    assert eng.load_scene("s1") is True
    assert eng.loaded_count() == 3
    assert eng.current_scene == "s1"
    assert eng.scenes["s1"].loaded is True
    for rid in ["model_a", "tex_b", "sfx_c"]:
        r = eng.get_resource(rid)
        assert r is not None and r.loaded and r.ref_count == 1


def test_failure_rolls_back_partial_load():
    """纹理失败时，已加载的模型必须回滚，不能成为孤儿资源。"""
    eng = _eng()
    assert eng.load_scene("s2") is False
    assert eng.loaded_count() == 0
    assert eng.get_resource("model_a") is None  # 引用归零已立即释放
    assert eng.scenes["s2"].loaded is False
    assert eng.current_scene is None


def test_failure_rollback_decrements_shared_ref():
    """回滚只减本次事务加的引用，不释放其他场景仍在用的共享资源。"""
    eng = _eng()
    eng.load_scene("s1")  # model_a ref=1
    assert eng.load_scene("s2") is False  # model_a 缓存命中 ref=2，失败后回滚
    r = eng.get_resource("model_a")
    assert r is not None and r.loaded and r.ref_count == 1
    assert eng.loaded_count() == 3  # s1 的资源完好


def test_cancel_load_rolls_back():
    """加载中途取消必须回滚已加载部分。"""
    eng = _eng()
    eng.add_scene(Scene("s5", ["r1", "r2", "r3"]))
    loaded_once = []
    def cancel_after_first(rid):
        loaded_once.append(rid)
        eng.cancel_load()
    eng.on_resource_loaded = cancel_after_first
    assert eng.load_scene("s5") is False
    assert len(loaded_once) == 1  # 只加载了 r1 就被取消
    assert eng.loaded_count() == 0
    assert eng.get_resource("r1") is None
    assert "cancelled" in eng.last_error


def test_ref_count_precise_no_leak_no_double_free():
    eng = _eng()
    eng.load_scene("s1")
    eng.add_scene(Scene("s3", ["model_a", "tex_b"]))
    eng.load_scene("s3")
    assert eng.get_resource("model_a").ref_count == 2
    eng.unload_scene("s1")
    assert eng.get_resource("model_a").ref_count == 1
    eng.unload_scene("s1")  # 重复卸载不得重复释放
    assert eng.get_resource("model_a").ref_count == 1
    eng.unload_scene("s3")
    assert eng.get_resource("model_a") is None  # 归零立即释放
    assert eng.loaded_count() == 0


def test_repeat_load_uses_cache_no_reload():
    """重复加载同一场景/共享资源命中缓存，不重复加载。"""
    eng = _eng()
    eng.load_scene("s1")
    assert eng.load_counts["model_a"] == 1
    assert eng.load_scene("s1") is True  # 重复加载同一场景
    assert eng.load_counts["model_a"] == 1  # 未重复加载
    assert eng.get_resource("model_a").ref_count == 1  # 引用计数未膨胀
    eng.add_scene(Scene("s3", ["model_a", "other"]))
    eng.load_scene("s3")
    assert eng.load_counts["model_a"] == 1  # 缓存命中
    assert eng.get_resource("model_a").ref_count == 2


def test_switch_scene_unloads_old_first():
    eng = _eng()
    eng.load_scene("s1")
    eng.add_scene(Scene("s3", ["model_d"]))
    assert eng.switch_scene("s3") is True
    assert eng.current_scene == "s3"
    assert eng.scenes["s1"].loaded is False
    assert eng.get_resource("tex_b") is None  # 旧场景资源已释放
    assert eng.get_resource("model_d").ref_count == 1


def test_zero_ref_released_immediately():
    eng = _eng()
    eng.load_scene("s1")
    eng.unload_scene("s1")
    assert eng.loaded_count() == 0
    assert eng.get_resource("model_a") is None
    assert eng.get_resource("tex_b") is None
    assert eng.get_resource("sfx_c") is None


def test_dependencies_auto_loaded():
    """A 依赖 B：加载 A 自动加载 B，卸载时一并释放。"""
    eng = _eng()
    eng.set_dependencies("model_a", ["skeleton", "anim"])
    eng.add_scene(Scene("s6", ["model_a"]))
    assert eng.load_scene("s6") is True
    assert eng.get_resource("skeleton").ref_count == 1
    assert eng.get_resource("anim").ref_count == 1
    assert eng.get_resource("model_a").ref_count == 1
    eng.unload_scene("s6")
    assert eng.get_resource("skeleton") is None
    assert eng.get_resource("model_a") is None


def test_progress_accurate():
    eng = _eng()
    eng.add_scene(Scene("s7", ["p1", "p2", "fail_p", "p4"]))
    seen = []
    eng.on_resource_loaded = lambda rid: seen.append(eng.load_progress())
    assert eng.load_scene("s7") is False
    assert seen == [(1, 4), (2, 4)]  # 进度准确反映已加载/总资源
    assert eng.load_progress() == (0, 4)  # 回滚后进度归零
    eng2 = _eng()
    assert eng2.load_scene("s1") is True
    assert eng2.load_progress() == (3, 3)


def test_failure_reports_clear_error():
    eng = _eng()
    assert eng.load_scene("s2") is False
    assert eng.last_error is not None
    assert "fail_tex" in eng.last_error
    assert "s2" in eng.last_error
    assert any("fail_tex" in e for e in eng.load_errors)
    eng2 = _eng()
    assert eng2.load_scene("nope") is False
    assert "nope" in eng2.last_error


def test_unload_releases_and_reload_works():
    """卸载后资源释放，重新加载能干净地重建。"""
    eng = _eng()
    eng.load_scene("s1")
    eng.unload_scene("s1")
    assert eng.loaded_count() == 0
    assert eng.load_scene("s1") is True
    assert eng.load_counts["model_a"] == 2  # 已释放，需重新加载
    assert eng.get_resource("model_a").ref_count == 1
