import pytest

from experiments.t2t.serving_slots.fixed_request_comparison import collect,compare_live,render


def test_same_fixed_offer_does_not_pool_hardware_or_contaminate_efficiency():
    rows=collect()
    at={(r["model"],r["cohort"],r["budget"],r["method"]):r for r in rows}
    capped=at["gemma","frozen90",128,"IBS capped"]
    assert capped["token_efficiency"]==pytest.approx(30814/209763)
    assert capped["F"]==2055 and capped["peak_decoder_Q"]=="128"
    assert at["gemma","frozen90",128,"IBS nominal"]["F"]==2042
    assert at["llada21","frozen90",32,"Native"]["F"]!=at["llada21","frozen90",32,"Flow W1"]["F"]
    w1=at["llada21","frozen90",32,"Flow W1"]
    assert (w1["N"],w1["F"],w1["history_Q"],w1["correct"])==(39520,7138,228416,75)
    assert w1["token_efficiency"]==pytest.approx(39520/228416)
    assert at["llada21","original30",32,"Flow W1"]["history_Q"]==71648
    assert at["llada21","new60",32,"Flow W1"]["history_Q"]==156768
    assert at["llada21","frozen90",64,"Flow W2"]["token_efficiency"]==pytest.approx(41553/358016)
    w2=at["gemma","frozen90",512,"Flow W2"]
    assert (w2["F"],w2["N"],w2["history_Q"])==(1494,32504,720896)
    assert w2["token_efficiency"]==pytest.approx(32504/720896)
    gemma_w1=at["gemma","frozen90",256,"Flow W1"]
    assert (gemma_w1["N"],gemma_w1["F"],gemma_w1["history_Q"],gemma_w1["correct"])==(31352,1397,336896,85)
    assert ("llada21","frozen90",64,"IBS capped") not in at
    assert "separately verified" in render(rows)


def test_active_window_is_its_own_sealed_arm_never_a_cherry_picked_composite():
    rows=collect()
    at={(r["model"],r["cohort"],r["budget"],r["method"]):r for r in rows}
    for q in (4,8,16):
        arm=[at["llada21",c,q,"Active window"] for c in ("original30","new60","frozen90")]
        for k in ("N","F","history_Q","correct","n"):
            assert arm[0][k]+arm[1][k]==arm[2][k]
        assert arm[2]["token_efficiency"]==pytest.approx(arm[2]["N"]/arm[2]["history_Q"])
    assert at["llada21","frozen90",4,"Active window"]["correct"]==72
    assert at["llada21","frozen90",8,"Active window"]["correct"]==72
    assert at["llada21","new60",16,"Active window"]["correct"]==49
    assert not any(r["method"]=="Active window" and (r["model"]!="llada21" or r["budget"]>16) for r in rows)
    repeat=at["llada21","frozen90",64,"Paid repeat"]
    assert repeat["correct"]==75 and repeat["token_efficiency"]==pytest.approx(43387/399051)
    assert len([r for r in rows if r["method"]=="Paid repeat"])==3
    paid=[r for r in rows if r["method"]=="Paid MASK"]
    assert {(r["budget"],r["model"]) for r in paid}=={(8,"llada21"),(16,"llada21")} and len(paid)==6
    assert at["llada21","frozen90",8,"Paid MASK"]["correct"]==70
    q16=at["llada21","frozen90",16,"Paid MASK"]
    assert q16["correct"]==75 and q16["macro_TPF"]==pytest.approx(3.35629259)
    assert q16["token_efficiency"]==pytest.approx(44943/175591)


def test_live_series_matches_named_dispatch_not_best_per_metric():
    rows=collect()
    at={(r["model"],r["cohort"],r["budget"],r["method"]):r for r in rows}
    for model,c,q,name in at:
        if name!="Live profile":
            continue
        arm={4:"Active window",8:"Active window",16:"Paid MASK",32:"Paid frontier",64:"Paid repeat"}[q] if model=="llada21" else "Request"
        assert at[model,c,q,name]=={**at[model,c,q,arm],"method":name}
    assert len([r for r in rows if r["method"]=="Live profile"])==30


def test_tradeoffs_compare_the_same_named_baseline_without_pooling_subgroups():
    table=compare_live(collect())
    ties=next(s for s in table.splitlines() if s.startswith("| gemma | 512 | Flow W2"))
    assert "| -1 / +1 / +0 |" in ties and "| +2.91 | no / no / yes |" in ties
    row=next(s for s in table.splitlines() if s.startswith("| llada21 | 4 | IBS capped"))
    assert "| +2 / +1 / +3 |" in row and row.endswith("yes / yes / yes |")
    assert table.count("| IBS capped |")==8 and table.count("| Flow W2 |")==2


def test_contiguous_frontier_reports_negative_top_result_alongside_q32():
    results=collect()
    selected={r["budget"]:r for r in results if r["cohort"]=="frozen90" and r["method"]=="Paid frontier"}
    assert set(selected)=={32,64}
    assert [selected[q]["correct"] for q in (32,64)]==[75,68]
    assert selected[32]["history_Q"]==231667


def test_primary_table_separates_live_from_experimental_arms():
    rows=collect()
    main=render(rows); experimental=render(rows,ablations=True)
    assert all("| Paid frontier |" not in line for line in main.splitlines())
    assert main.count("| Live profile | IBS capped | IBS nominal | Native | Flow W1 | Flow W2 |")==3
    assert experimental.count("| Paid repeat | Paid frontier |")==3
    assert not any(line.startswith("| gemma |") for line in experimental.splitlines())
