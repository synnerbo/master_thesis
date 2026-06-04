# feature_combination.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureFlags:
    # RV family
    include_rv: bool
    include_rv_w: bool
    include_rv_m: bool

    # realized “components”
    include_cc: bool
    include_jc: bool
    include_pv: bool
    include_nv: bool
    include_sj: bool
    include_cq: bool

    # IV
    include_iv_d1: bool
    include_iv_w1: bool
    include_iv_m1: bool

    # IV term structure
    include_iv_m3: bool
    include_iv_m6: bool
    include_iv_y1: bool

    include_iv_slope_d1_w1: bool
    include_iv_slope_d1_m1: bool
    include_iv_slope_w1_m1: bool
    include_iv_slope_m1_m3: bool
    include_iv_slope_m3_m6: bool
    include_iv_slope_m1_y1: bool
    include_iv_slope_m6_y1: bool

    include_iv_curve_d1_w1_m1: bool
    include_iv_curve_w1_m1_m3: bool
    include_iv_curve_m1_m6_y1: bool
    include_iv_curve_m3_m6_y1: bool

    # RR
    include_rr_d1: bool
    include_rr_w1: bool
    include_rr_m1: bool

    # BF
    include_bf_d1: bool
    include_bf_w1: bool
    include_bf_m1: bool


def _all_false() -> FeatureFlags:
    return FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,

        include_iv_d1=False, include_iv_w1=False, include_iv_m1=False,

        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False,
        include_iv_slope_d1_m1=False,
        include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False,
        include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False,
        include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False,
        include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False,
        include_iv_curve_m3_m6_y1=False,

        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    )


VERSION_TO_FLAGS: dict[str, FeatureFlags] = {
    # RV: RV, RV_W, RV_M
    "RV": FeatureFlags(
        include_rv=True, include_rv_w=True, include_rv_m=True,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=False, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV: IV_D1, IV_W1, IV_M1
    "IV": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_RV: RV, RV_W, RV_M, IV_*
    "IV_RV": FeatureFlags(
        include_rv=True, include_rv_w=True, include_rv_m=True,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),


    "IV_RV_D": FeatureFlags(
        include_rv=True, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    "IV_RV_W": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=False, include_iv_w1=True, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # RV_CJ: CC, JC, RV_W, RV_M
    "RV_CJ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=True, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=False, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # RV_CQ: CC, CQ, RV_W, RV_M
    "RV_CQ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=True,
        include_iv_d1=False, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # RV_SV: PV, NV, RV_W, RV_M
    "RV_SV": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=False, include_jc=False, include_pv=True, include_nv=True, include_sj=False, include_cq=False,
        include_iv_d1=False, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # RV_SJ: CC, SJ, RV_W, RV_M
    "RV_SJ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=False, include_pv=False, include_nv=False, include_sj=True, include_cq=False,
        include_iv_d1=False, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_CJ: CC, JC, RV_W, RV_M, IV_*
    "IV_CJ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=True, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_CQ: CC, CQ, RV_W, RV_M, IV_*
    "IV_CQ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=True,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_SV: PV, NV, RV_W, RV_M, IV_*
    "IV_SV": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=False, include_jc=False, include_pv=True, include_nv=True, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_SJ: CC, SJ, RV_W, RV_M, IV_*
    "IV_SJ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=False, include_pv=False, include_nv=False, include_sj=True, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_RR: IV_* + RR_*
    "IV_RR": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=True, include_rr_w1=True, include_rr_m1=True,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_BF: IV_* + BF_*
    "IV_BF": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=True, include_bf_w1=True, include_bf_m1=True,
    ),

    # IV_BF_RR: IV_* + RR_* + BF_*
    "IV_BF_RR": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=True, include_rr_w1=True, include_rr_m1=True,
        include_bf_d1=True, include_bf_w1=True, include_bf_m1=True,
    ),

    # IV_slope
    "IV_SLOPE": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=False, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
        # legge til d1_m1 slope
    ),

    # IV_curve
    "IV_CURVE": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=False, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve
    "IV_SLOPE_CURVE": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve
    "IV_SLOPE_CURVE_D": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=False, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    "IV_SLOPE_CURVE_W": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=False, include_iv_w1=True, include_iv_m1=False,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve_CJ
    "IV_SLOPE_CURVE_CJ": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=True, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve_SV
    "IV_SLOPE_CURVE_SV": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=False, include_jc=False, include_pv=True, include_nv=True, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve_CJ_SV
    "IV_SLOPE_CURVE_CJ_SV": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=True, include_pv=True, include_nv=True, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve
    "IV_SLOPE_CURVE_RR": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=True, include_rr_w1=True, include_rr_m1=True,
        include_bf_d1=False, include_bf_w1=False, include_bf_m1=False,
    ),

    # IV_slope_curve
    "IV_SLOPE_CURVE_BF": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=False, include_rr_w1=False, include_rr_m1=False,
        include_bf_d1=True, include_bf_w1=True, include_bf_m1=True,
    ),

    # IV_slope_curve
    "IV_SLOPE_CURVE_RR_BF": FeatureFlags(
        include_rv=False, include_rv_w=False, include_rv_m=False,
        include_cc=False, include_jc=False, include_pv=False, include_nv=False, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=True, include_rr_w1=True, include_rr_m1=True,
        include_bf_d1=True, include_bf_w1=True, include_bf_m1=True,
    ),

    # IV_slope_curve
    "IV_SLOPE_CURVE_CJ_SV_RR_BF": FeatureFlags(
        include_rv=False, include_rv_w=True, include_rv_m=True,
        include_cc=True, include_jc=True, include_pv=True, include_nv=True, include_sj=False, include_cq=False,
        include_iv_d1=True, include_iv_w1=True, include_iv_m1=True,
        include_iv_m3=False, include_iv_m6=False, include_iv_y1=False,
        include_iv_slope_d1_w1=False, include_iv_slope_d1_m1=True, include_iv_slope_w1_m1=False,
        include_iv_slope_m1_m3=False, include_iv_slope_m3_m6=False,
        include_iv_slope_m1_y1=False, include_iv_slope_m6_y1=False,
        include_iv_curve_d1_w1_m1=True, include_iv_curve_w1_m1_m3=False,
        include_iv_curve_m1_m6_y1=False, include_iv_curve_m3_m6_y1=False,
        include_rr_d1=True, include_rr_w1=True, include_rr_m1=True,
        include_bf_d1=True, include_bf_w1=True, include_bf_m1=True,
    ),

}


def _get_feature_flags(version: str) -> FeatureFlags:
    key = version.strip().upper().replace("-", "_")
    if key not in VERSION_TO_FLAGS:
        raise ValueError(f"Unknown VERSION '{version}'. Valid: {list(VERSION_TO_FLAGS.keys())}")
    return VERSION_TO_FLAGS[key]

def get_feature_cols(version: str) -> list[str]:
    """
    Returns the exact list of feature column names to use in df,
    implied by VERSION, in a fixed deterministic order.
    """
    f = _get_feature_flags(version)

    cols: list[str] = []

    if f.include_rv:   cols.append("RV")
    if f.include_rv_w: cols.append("RV_W")
    if f.include_rv_m: cols.append("RV_M")

    if f.include_cc: cols.append("CC")
    if f.include_jc: cols.append("JC")
    if f.include_pv: cols.append("PV")
    if f.include_nv: cols.append("NV")
    if f.include_sj: cols.append("SJ")
    if f.include_cq: cols.append("CQ")

    if f.include_iv_d1: cols.append("IV_D1")
    if f.include_iv_w1: cols.append("IV_W1")
    if f.include_iv_m1: cols.append("IV_M1")
    if f.include_iv_m3: cols.append("IV_M3")
    if f.include_iv_m6: cols.append("IV_M6")
    if f.include_iv_y1: cols.append("IV_Y1")

    if f.include_iv_slope_d1_w1: cols.append("IV_SLOPE_D1_W1")
    if f.include_iv_slope_d1_m1: cols.append("IV_SLOPE_D1_M1")
    if f.include_iv_slope_w1_m1: cols.append("IV_SLOPE_W1_M1")
    if f.include_iv_slope_m1_m3: cols.append("IV_SLOPE_M1_M3")
    if f.include_iv_slope_m3_m6: cols.append("IV_SLOPE_M3_M6")
    if f.include_iv_slope_m1_y1: cols.append("IV_SLOPE_M1_Y1")
    if f.include_iv_slope_m6_y1: cols.append("IV_SLOPE_M6_Y1")

    if f.include_iv_curve_d1_w1_m1: cols.append("IV_CURVE_D1_W1_M1")
    if f.include_iv_curve_w1_m1_m3: cols.append("IV_CURVE_W1_M1_M3")
    if f.include_iv_curve_m1_m6_y1: cols.append("IV_CURVE_M1_M6_Y1")
    if f.include_iv_curve_m3_m6_y1: cols.append("IV_CURVE_M3_M6_Y1")

    if f.include_rr_d1: cols.append("RR_D1")
    if f.include_rr_w1: cols.append("RR_W1")
    if f.include_rr_m1: cols.append("RR_M1")

    if f.include_bf_d1: cols.append("BF_D1")
    if f.include_bf_w1: cols.append("BF_W1")
    if f.include_bf_m1: cols.append("BF_M1")

    if not cols:
        raise ValueError(f"VERSION '{version}' produced empty feature set")

    return cols