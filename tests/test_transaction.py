"""事务性场景加载不变量测试（>=12 个断言）。

覆盖：正常加载、失败回滚、取消回滚、引用计数精确、资源缓存、
场景切换、引用零释放、依赖加载、进度准确、失败错误信息、
重复加载不重复、卸载后资源释放。
"""

from __future__ import annotations

import asyncio

from src import CancelToken, Scene, SceneLoader


def _eng() -> SceneLoader:
    eng = SceneLoader()
    eng.add_scene(Scene("s1", ["model_a", "tex_b", "sfx_c"]))
    eng.add_scene(Scene("s2", ["model_a", "fail_x"]))
    return eng


# 1. 正常加载 ---------------------------------------------------------- #
def test_normal_load_commits_all():
    eng = _eng()
    assert eng.load_scene("s1") is True
    assert eng.loaded_count() == 3
    assert eng.current_scene == "s1"
    assert eng.scenes["s1"].loaded is True
    for rid in ("model_a", "tex_b", "sfx_c"):
        assert eng.get_resource(rid).loaded is True


# 2. 失败回滚：失败前已加载的资源必须全部回滚 --------------------------- #
def test_failure_rolls_back_partial_load():
    eng = _eng()
    assert eng.load_scene("s2") is False
    r = eng.get_resource("model_a")
    assert r is None or (not r.loaded and r.ref_count == 0)
    assert eng.loaded_count() == 0
    assert eng.scenes["s2"].loaded is False
    assert eng.current_scene is None
    # 失败的资源不应留下任何状态
    assert eng.get_resource("fail_x") is None or not eng.get_resource("fail_x").loaded


# 10. 失败必须有明确错误信息，不能静默 --------------------------------- #
def test_failure_records_explicit_error():
    eng = _eng()
    eng.load_scene("s2")
    assert eng.last_error is not None
    assert "fail_x" in eng.last_error
    assert "s2" in eng.last_error
    assert "rolled back" in eng.last_error
    assert eng.load_scene("missing") is False
    assert "missing" in eng.last_error


# 4. 引用计数精确 ------------------------------------------------------ #
def test_ref_counts_precise_across_share_and_unload():
    eng = _eng()
    eng.add_scene(Scene("s3", ["model_a", "other"]))
    assert eng.load_scene("s1") is True
    assert eng.load_scene("s3") is True
    shared = eng.get_resource("model_a")
    assert shared.ref_count == 2
    assert eng.get_resource("tex_b").ref_count == 1
    eng.unload_scene("s1")
    assert shared.ref_count == 1 and shared.loaded  # s3 仍持有
    assert eng.get_resource("tex_b").ref_count == 0
    assert not eng.get_resource("tex_b").loaded     # 引用归零立即释放
    eng.unload_scene("s3")
    assert shared.ref_count == 0 and not shared.loaded
    # 卸载未知 / 重复卸载不能把计数减成负数
    eng.unload_scene("s3")
    eng.unload_scene("ghost")
    assert shared.ref_count == 0


# 5 & 11. 资源缓存：重复加载不重复真正加载 ----------------------------- #
def test_cached_resource_not_fetched_twice():
    eng = _eng()
    eng.add_scene(Scene("s3", ["model_a", "other"]))
    eng.load_scene("s1")
    eng.load_scene("s3")
    assert eng.fetch_count("model_a") == 1  # 第二次是缓存命中
    assert eng.fetch_count("tex_b") == 1
    # 已加载场景重复加载：不重复获取任何资源
    before = {rid: r.ref_count for rid, r in eng.resources.items()}
    assert eng.load_scene("s1") is True
    after = {rid: r.ref_count for rid, r in eng.resources.items()}
    assert before == after
    assert eng.fetch_count("model_a") == 1


# 12. 卸载后资源释放 --------------------------------------------------- #
def test_unload_releases_all_scene_resources():
    eng = _eng()
    eng.load_scene("s1")
    eng.unload_scene("s1")
    assert eng.loaded_count() == 0
    for rid in ("model_a", "tex_b", "sfx_c"):
        r = eng.get_resource(rid)
        assert r is not None
        assert r.ref_count == 0 and not r.loaded
    assert eng.current_scene is None


# 6. 场景切换：先卸载旧场景再加载新场景 --------------------------------- #
def test_switch_unloads_old_before_loading_new():
    eng = _eng()
    eng.add_scene(Scene("s3", ["model_d"]))
    eng.load_scene("s1")
    assert eng.switch_scene("s3") is True
    assert eng.current_scene == "s3"
    assert eng.scenes["s1"].loaded is False
    assert eng.scenes["s3"].loaded is True
    for rid in ("model_a", "tex_b", "sfx_c"):
        r = eng.get_resource(rid)
        assert r.ref_count == 0 and not r.loaded
    assert eng.get_resource("model_d").loaded is True


def test_switch_to_failing_scene_keeps_old_unloaded():
    eng = _eng()
    eng.load_scene("s1")
    assert eng.switch_scene("s2") is False  # 旧场景先卸载，新场景事务失败回滚
    assert eng.scenes["s1"].loaded is False
    assert eng.loaded_count() == 0


# 8. 资源依赖：A 依赖 B，加载 A 时自动加载 B ---------------------------- #
def test_dependency_loaded_automatically():
    eng = _eng()
    eng.add_dependency("model_a", "tex_dep")
    eng.add_scene(Scene("sd", ["model_a"]))
    required = eng.required_resources("sd")
    assert required == ["tex_dep", "model_a"]  # 依赖在前
    assert eng.load_scene("sd") is True
    assert eng.get_resource("tex_dep").loaded
    assert eng.get_resource("model_a").ref_count == 1
    eng.unload_scene("sd")  # 卸载也必须释放依赖
    assert not eng.get_resource("tex_dep").loaded
    assert eng.get_resource("tex_dep").ref_count == 0
    assert eng.get_resource("model_a").ref_count == 0


def test_transitive_dependency_failure_rolls_back():
    eng = _eng()
    eng.add_dependency("a", "b")
    eng.add_dependency("b", "fail_c")
    eng.add_scene(Scene("sc", ["a"]))
    assert eng.load_scene("sc") is False
    assert eng.loaded_count() == 0
    assert "fail_c" in eng.last_error
    assert not eng.scenes["sc"].loaded


# 9. 进度准确：已加载 / 总资源（含依赖） -------------------------------- #
def test_progress_reflects_loaded_over_total():
    eng = _eng()
    events = []
    assert eng.load_scene("s1", on_progress=lambda d, t: events.append((d, t)))
    assert events == [(1, 3), (2, 3), (3, 3)]

    eng2 = _eng()
    eng2.add_dependency("model_a", "tex_dep")
    events2 = []
    eng2.load_scene("s1", on_progress=lambda d, t: events2.append((d, t)))
    assert events2[-1] == (4, 4)
    assert events2[0] == (1, 4)
    assert [d for d, _ in events2] == [1, 2, 3, 4]


def test_progress_on_failure_and_cached_reload():
    eng = _eng()
    events = []
    eng.load_scene("s2", on_progress=lambda d, t: events.append((d, t)))
    assert events[-1][1] == 2          # 总数准确
    assert events[-1][0] == 1          # 仅 1 个已加载随即回滚
    # 已加载场景重复回调进度按总量报告
    eng.load_scene("s1")
    events2 = []
    eng.load_scene("s1", on_progress=lambda d, t: events2.append((d, t)))
    assert events2 == [(3, 3)]


# 3. 取消加载（异步）必须回滚已加载部分 --------------------------------- #
def test_async_cancel_before_start_rolls_back():
    eng = _eng()
    eng.add_scene(Scene("sx", ["r1", "r2", "r3"]))
    token = CancelToken()
    token.cancel()
    result = asyncio.run(eng.load_scene_async("sx", cancel_token=token))
    assert result is False
    assert eng.loaded_count() == 0
    assert "cancelled" in eng.last_error


def test_async_cancel_midflight_rolls_back_partial():
    eng = _eng()
    eng.add_scene(Scene("sx", ["r1", "r2", "r3"]))
    token = CancelToken()

    def cancel_after_first(done: int, total: int) -> None:
        if done == 1:
            token.cancel()

    result = asyncio.run(
        eng.load_scene_async(
            "sx",
            cancel_token=token,
            on_progress=cancel_after_first,
            step_delay=0.001,
        )
    )
    assert result is False
    r1 = eng.get_resource("r1")
    assert r1 is None or (not r1.loaded and r1.ref_count == 0)
    r2 = eng.get_resource("r2")
    assert r2 is None or not r2.loaded
    assert eng.loaded_count() == 0
    assert "cancelled" in eng.last_error
    assert not eng.scenes["sx"].loaded


def test_async_normal_load_matches_sync_invariants():
    eng = _eng()
    events = []
    result = asyncio.run(
        eng.load_scene_async("s1", on_progress=lambda d, t: events.append((d, t)))
    )
    assert result is True
    assert events == [(1, 3), (2, 3), (3, 3)]
    assert all(r.ref_count == 1 for r in eng.resources.values())
