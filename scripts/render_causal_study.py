"""Render question-led scientific figures and the Chinese Markdown review."""

import json
from pathlib import Path
import pathlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullLocator, NullFormatter, FuncFormatter
import numpy as np
import pandas as pd

from src.policy import select_under_budget
from solutions.causal_solution import qini_curve
from solutions.causal_study import METHODS

OUT = ROOT / "artifacts/causal_study_20261015"
FIG = OUT / "figures"
COLORS = dict(zip(METHODS,["#8055A5","#2479AE","#D47720","#21866E"]))
STYLES = dict(zip(METHODS,["-.","-","--",":"]))
LABELS = {"S-logistic":"S: logistic + interactions", "T-logistic":"T: logistic",
          "T-boosting":"T: gradient boosting", "CausalForestDML":"Causal Forest DML"}
SCENES = {"smooth":"Smooth response", "nonlinear":"Thresholds + interactions", "hidden_confounder":"Hidden confounder"}


def save(fig,name):
    fig.savefig(FIG/f"{name}.png",dpi=175,bbox_inches="tight",facecolor="white")
    fig.savefig(FIG/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=0.16)
    ax.set_axisbelow(True)


def box(ax,x,y,w,h,text,color="#E8EFF5",size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=0.05",
                              facecolor=color,edgecolor="#A9B8C5",linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size)


def arrow(ax,a,b,color="#52687A",style="-|>"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=15,color=color,linewidth=1.5))


def mechanisms():
    fig,axes=plt.subplots(1,3,figsize=(15,6.4),layout="constrained")
    titles=["S-Learner: share an outcome model","T-Learner: separate outcome models","Causal Forest: target effect variation"]
    for ax,title in zip(axes,titles):
        ax.set(xlim=(0,5),ylim=(0,7)); ax.axis("off"); ax.set_title(title,loc="left",fontweight="bold",fontsize=12,pad=14)
    ax=axes[0]
    box(ax,.4,5.6,4.1,.8,"One shared model: f(X, T, X*T)","#EDE6F3")
    box(ax,.1,3.9,2.1,.9,"Same X\nset T = 0"); box(ax,2.7,3.9,2.1,.9,"Same X\nset T = 1")
    arrow(ax,(2.0,5.6),(1.2,4.8)); arrow(ax,(3.0,5.6),(3.7,4.8))
    box(ax,.6,2.0,3.8,1.0,"Effect = f(X,1) - f(X,0)","#EDE6F3")
    arrow(ax,(1.2,3.9),(1.9,3)); arrow(ax,(3.7,3.9),(3.1,3))
    ax.text(2.5,.75,"Interactions determine flexibility.\nSharing may stabilize or restrict estimates.",ha="center",va="center",fontsize=10)
    ax=axes[1]
    box(ax,.1,5.1,2.1,1.1,"Control arm\nmu0(X)","#E0EDF6"); box(ax,2.7,5.1,2.1,1.1,"Treated arm\nmu1(X)","#E0EDF6")
    box(ax,.6,2.6,3.8,1.0,"Effect = mu1(X) - mu0(X)","#E0EDF6")
    arrow(ax,(1.2,5.1),(1.9,3.6)); arrow(ax,(3.7,5.1),(3.1,3.6))
    ax.text(2.5,.9,"Each model sees fewer observations.\nTwo prediction errors enter the difference.",ha="center",va="center",fontsize=10)
    ax=axes[2]
    box(ax,.1,5.5,4.7,1.0,"Cross-fitted outcome + treatment models\nY residual and T residual", "#E0F0E8",10)
    box(ax,.1,3.6,4.7,1.0,"Forest finds local effect differences\nHonesty separates split / leaf estimation", "#E0F0E8",10)
    box(ax,.1,1.7,4.7,1.0,r"Local moment: E[T~ (Y~ - tau(X) T~) | X] = 0", "#E0F0E8",9)
    arrow(ax,(2.5,5.5),(2.5,4.6)); arrow(ax,(2.5,3.6),(2.5,2.7))
    ax.text(2.5,.5,"Requires identification assumptions too.\nUnmeasured confounding remains a problem.",ha="center",va="center",fontsize=10)
    save(fig,"01_effect_estimators")


def learning_curves(main,metric,ylabel,name):
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.5),layout="constrained")
    for ax,(scenario,title) in zip(axes,SCENES.items()):
        part=main[main.scenario==scenario]
        for method in METHODS:
            grouped=part[part.method==method].groupby("training_rows")[metric].agg(["mean","min","max"])
            x=grouped.index.to_numpy()
            ax.plot(x,grouped["mean"],marker="o",color=COLORS[method],ls=STYLES[method],label=LABELS[method])
            ax.fill_between(x,grouped["min"],grouped["max"],color=COLORS[method],alpha=.10)
        if metric=="true_policy_profit":
            ref=part.truth_ranked_reference_profit.iloc[0]
            ax.axhline(ref,color="#545454",ls="--",lw=1,label="Truth-ranked reference")
            ax.axhline(0,color="black",lw=.8)
        style(ax); ax.set_xscale("log"); ax.set_xticks([2_000,10_000,50_000],["2k","10k","50k"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title,fontweight="bold"); ax.set_xlabel("Training customers")
    axes[0].set_ylabel(ylabel)
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=9)
    fig.suptitle("Lines: mean over five training seeds; bands: min–max (not confidence intervals)",fontsize=11)
    save(fig,name)


def hidden_audit(metrics):
    sub=metrics[(metrics.scenario=="hidden_confounder")&(metrics.training_rows==50_000)]
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout="constrained")
    for ax,metric,ylabel in zip(axes,["ate_bias","true_policy_profit"],["ATE bias (probability points)","True expected incremental profit ($)"]):
        for i,method in enumerate(METHODS):
            for seed in sorted(sub.seed.unique()):
                pair=sub[(sub.method==method)&(sub.seed==seed)].sort_values("reveal_u")
                offset=(seed-42)*.025
                x=np.array([i-.16,i+.16])+offset
                y=pair[metric].to_numpy()
                ax.plot(x,y,color=COLORS[method],alpha=.45,lw=1)
                ax.scatter(x[0],y[0],facecolor="white",edgecolor=COLORS[method],s=35,zorder=4)
                ax.scatter(x[1],y[1],color=COLORS[method],s=35,zorder=4)
        ax.set_xticks(range(4),["S-logistic","T-logistic","T-boosting","Causal Forest"],rotation=12)
        ax.axhline(0,color="black",lw=.8); ax.set_ylabel(ylabel); style(ax)
        if metric=="ate_bias":
            ax.set_ylabel("ATE bias (percentage points)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{100*value:.0f}"))
    axes[0].set_title("Missing U → observing U",fontweight="bold")
    axes[1].set_title("Same data, same seed, same budget",fontweight="bold")
    fig.suptitle("50k training customers · open point: U hidden · filled point: U included · paired by seed",fontsize=11)
    save(fig,"04_hidden_variable_audit")


def prediction_file(scenario,method,n=10_000,seed=42,pilot=False):
    return OUT/"runs"/scenario/f"n{n}"/f"seed{seed}"/method/("pilot_predictions.csv.gz" if pilot else "predictions.csv.gz")


def qini_panels():
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout="constrained")
    for ax,(scenario,title) in zip(axes,SCENES.items()):
        pilot=pd.read_csv(OUT/"evaluation"/scenario/"pilot.csv.gz")
        for method in METHODS:
            pred=pd.read_csv(prediction_file(scenario,method,pilot=True))
            assert pilot.customer_id.equals(pred.customer_id)
            curve=qini_curve(pilot,pred.estimated_uplift.to_numpy())
            ax.plot(curve.share,curve.gain,color=COLORS[method],ls=STYLES[method],label=LABELS[method])
        ax.plot([0,1],[0,curve.gain.iloc[-1]],color="black",ls="--",lw=1,label="Random-targeting line")
        style(ax); ax.set_xlabel("Share of randomized pilot targeted"); ax.set_title(title,fontweight="bold")
    axes[0].set_ylabel("Estimated cumulative incremental purchases")
    fig.legend(*axes[-1].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=9)
    fig.suptitle("10k training customers · seed 42 · the same independent 5k randomized pilot per scenario",fontsize=11)
    save(fig,"05_randomized_pilot_qini")


def effect_maps():
    customers=pd.read_csv(OUT/"evaluation/nonlinear/customers.csv.gz")
    truth=pd.read_csv(OUT/"evaluation/nonlinear/truth.csv.gz")
    xb=pd.cut(customers.price_sensitivity,np.linspace(0,1,9),labels=False)
    yb=pd.cut(customers.offer_affinity,np.linspace(0,1,9),labels=False)
    arrays=[truth.true_cate.to_numpy()]
    titles=["Simulator truth"]
    for method in METHODS:
        arrays.append(pd.read_csv(prediction_file("nonlinear",method)).estimated_uplift.to_numpy())
        titles.append(LABELS[method].replace(": ",":\n"))
    grids=[]
    for values in arrays:
        grid=pd.DataFrame({"x":xb,"y":yb,"v":values}).pivot_table(index="y",columns="x",values="v",aggfunc="mean")
        grids.append(grid.reindex(index=range(8),columns=range(8)).to_numpy())
    scale=max(abs(np.nanmin(grids)),abs(np.nanmax(grids)))
    fig,axes=plt.subplots(1,5,figsize=(17,4),layout="constrained")
    for ax,grid,title in zip(axes,grids,titles):
        im=ax.imshow(grid,origin="lower",extent=[0,1,0,1],cmap="RdBu_r",norm=TwoSlopeNorm(0,-scale,scale))
        ax.axvline(.6,color="black",ls="--",lw=.8); ax.axhline(.5,color="black",ls="--",lw=.8)
        ax.set_title(title,fontsize=10); ax.set_xlabel("Price sensitivity")
    axes[0].set_ylabel("Offer affinity")
    fig.colorbar(im,ax=axes,label="Mean CATE / estimated uplift (probability change)",shrink=.8)
    fig.suptitle("Nonlinear scenario · 10k training rows · seed 42 · other covariates averaged within bins",fontsize=11)
    save(fig,"06_effect_surface")


def calibration():
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.5),layout="constrained")
    for ax,(scenario,title) in zip(axes,SCENES.items()):
        limits=[]
        for method in METHODS:
            pred=pd.read_csv(prediction_file(scenario,method))
            pred["decile"]=pd.qcut(pred.estimated_uplift.rank(method="first"),10,labels=False)
            agg=pred.groupby("decile")[["estimated_uplift","true_cate"]].mean()
            ax.plot(agg.estimated_uplift,agg.true_cate,marker="o",color=COLORS[method],ls=STYLES[method],label=LABELS[method])
            limits.extend([agg.min().min(),agg.max().max()])
        lo,hi=min(limits)-.01,max(limits)+.01
        ax.plot([lo,hi],[lo,hi],color="black",ls="--",lw=1); ax.set(xlim=(lo,hi),ylim=(lo,hi))
        style(ax); ax.set_xlabel("Mean predicted uplift within predicted decile"); ax.set_title(title,fontweight="bold")
    axes[0].set_ylabel("Mean true CATE in the same customers")
    fig.legend(*axes[-1].get_legend_handles_labels(),loc="outside lower center",ncol=2,fontsize=9)
    fig.suptitle("Does a predicted +20-point uplift mean a true +20-point effect? · 10k training rows · seed 42",fontsize=11)
    save(fig,"07_effect_calibration")


def budget_curves():
    customers=pd.read_csv(OUT/"evaluation/nonlinear/customers.csv.gz")
    truth=pd.read_csv(OUT/"evaluation/nonlinear/truth.csv.gz")
    cost=customers.expected_coupon_cost.to_numpy(); margin=customers.expected_order_margin.to_numpy()
    actual=truth.true_cate.to_numpy()*margin-cost
    budgets=np.linspace(0,4_000,41)
    fig,ax=plt.subplots(figsize=(10.5,4.8),layout="constrained")
    for method in METHODS:
        pred=pd.read_csv(prediction_file("nonlinear",method)).estimated_uplift.to_numpy()
        rank=pred*margin-cost
        values=[actual[select_under_budget(rank,cost,budget).astype(bool)].sum() for budget in budgets]
        ax.plot(budgets,values,color=COLORS[method],ls=STYLES[method],label=LABELS[method])
    ax.axvline(2_500,color="black",ls="--",lw=1,label="Study budget: $2,500")
    ax.axhline(0,color="black",lw=.8); style(ax)
    ax.set(xlabel="Available coupon budget ($)",ylabel="True expected incremental profit ($)",
           title="Would you still choose the same ranking at a different budget?")
    ax.legend(fontsize=9); save(fig,"08_budget_sensitivity")


def runtime_and_tradeoff(main):
    fig,axes=plt.subplots(1,2,figsize=(13.5,4.8),layout="constrained")
    ax=axes[0]
    for method in METHODS:
        sub=main[main.method==method]
        agg=sub.groupby("training_rows").fit_seconds.agg(["median","min","max"])
        ax.plot(agg.index,agg["median"],marker="o",color=COLORS[method],ls=STYLES[method],label=LABELS[method])
        ax.fill_between(agg.index,agg["min"],agg["max"],color=COLORS[method],alpha=.1)
    ax.set(xscale="log",yscale="log",xlabel="Training customers",ylabel="Fit wall time (seconds)",title="CPU cost of flexibility")
    ax.set_xticks([2_000,10_000,50_000],["2k","10k","50k"]); style(ax)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax=axes[1]
    markers={"smooth":"o","nonlinear":"s","hidden_confounder":"^"}
    for method in METHODS:
        for scene in SCENES:
            s=main[(main.method==method)&(main.scenario==scene)&(main.training_rows==10_000)]
            ax.scatter(s.pilot_qini,s.true_policy_profit,color=COLORS[method],marker=markers[scene],alpha=.7,s=40)
    ax.axhline(0,color="black",lw=.8); style(ax)
    ax.set(xlabel="Qini on independent randomized pilot",ylabel="True profit on fixed target population ($)",title="Ranking evidence and decision value")
    ax.text(.02,.98,"10k training rows\no smooth · square nonlinear · triangle hidden U",transform=ax.transAxes,va="top",fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,fontsize=9)
    save(fig,"09_cost_and_decision_tradeoff")


def hidden_dag():
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax,measured in zip(axes,[False,True]):
        ax.set(xlim=(0,10),ylim=(0,5)); ax.axis("off")
        box(ax,.3,1.1,2,.8,"Coupon T"); box(ax,7.6,1.1,2,.8,"Purchase Y")
        box(ax,3.8,3.4,2,.8,"U observed" if measured else "U hidden", "#D8EEE5" if measured else "#F6DEDA")
        box(ax,3.8,.1,2,.8,"Observed X")
        arrow(ax,(2.3,1.5),(7.6,1.5),"#2479AE")
        arrow(ax,(4.0,3.4),(1.6,1.9),"#B45745"); arrow(ax,(5.5,3.4),(8.4,1.9),"#B45745")
        arrow(ax,(3.8,.6),(2,1.1)); arrow(ax,(5.8,.6),(8,1.1))
        ax.set_title("Paired audit: include U in adjustment" if measured else "More rows preserve the open backdoor path",fontsize=12,fontweight="bold")
        ax.text(5,2.45,"T <- U -> Y is adjusted" if measured else "T <- U -> Y remains open",ha="center",color="#21866E" if measured else "#B45745",fontsize=12)
    save(fig,"10_hidden_confounding_dag")


def classroom_generalization():
    path=OUT/"classroom_analysis/model_check.csv"
    if not path.exists():
        return False
    frame=pd.read_csv(path).dropna(subset=["held-out AUC mu0"])
    train=frame[["training AUC mu0","training AUC mu1"]].mean(axis=1)
    heldout=frame[["held-out AUC mu0","held-out AUC mu1"]].mean(axis=1)
    names=["T-logistic","T-boosting","S-logistic"]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout="constrained")
    for i,(a,b) in enumerate(zip(train,heldout)):
        axes[0].plot([a,b],[i,i],color="#8899AA",linewidth=3)
        axes[0].scatter(a,i,facecolors="white",edgecolors="#2479AE",s=65,zorder=3)
        axes[0].scatter(b,i,color="#2479AE",s=65,zorder=3)
    axes[0].set(yticks=range(3),yticklabels=names,xlabel="Mean outcome-model AUC over the two arms",
                title="Open: training · filled: held-out")
    axes[0].set_xlim(.5,1); style(axes[0])
    axes[1].barh(names,frame["pilot Qini"],color=[COLORS[m] for m in ["T-logistic","T-boosting","S-logistic"]])
    axes[1].set(xlabel="Qini on independent randomized pilot",title="Does the outcome fit translate into uplift ranking?")
    style(axes[1]); fig.suptitle("Classroom data · 7k train / 3k held-out · independent 5k pilot",fontsize=11)
    save(fig,"11_generalization_and_uplift")
    return True


def markdown_table(frame):
    cols=list(frame.columns)
    rows=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    rows.extend("| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(rows)


def report(metrics,manifest):
    main=metrics[~metrics.reveal_u]
    agg=main.groupby(["scenario","training_rows","method"],as_index=False).agg(
        rmse=("cate_rmse","mean"),profit=("true_policy_profit","mean"),bias=("ate_bias","mean"),fit=("fit_seconds","median"))
    at50=agg[agg.training_rows==50_000]
    winners=[]
    for scene in SCENES:
        s=at50[at50.scenario==scene]
        best=s.loc[s.rmse.idxmin()]; profitable=s.loc[s.profit.idxmax()]
        winners.append(f"- **{SCENES[scene]}**：50k 训练行时，平均 CATE RMSE 最低的是 {best.method}（{best.rmse:.4f}）；{"平均策略收益最高" if profitable.profit > 0 else "所有方法均亏损，亏损最少"}的是 {profitable.method}（{profitable.profit:,.2f}）。")
    h=metrics[(metrics.scenario=="hidden_confounder")&(metrics.training_rows==50_000)&(metrics.method=="T-logistic")]
    hidden=h[~h.reveal_u].ate_bias.mean(); observed=h[h.reveal_u].ate_bias.mean()
    rows=[]
    for r in agg[agg.training_rows==10_000].itertuples():
        rows.append({"场景":SCENES[r.scenario],"方法":r.method,"CATE RMSE":f"{r.rmse:.4f}","真实期望增量利润":f"{r.profit:,.2f}","训练秒数（中位）":f"{r.fit:.3f}"})
    def image(name):
        return f"![{name}]("+(FIG/f"{name}.png").relative_to(OUT).as_posix()+")"
    def link(path,label):
        return f"[{label}]("+pathlib.PurePath(path).as_posix()+")"
    execution=json.loads((OUT/"notebook_execution.json").read_text(encoding="utf-8")) if (OUT/"notebook_execution.json").exists() else []
    execution_lines="\n".join(
        f"- {link('executed/'+Path(row['source']).stem+'.executed.ipynb',Path(row['source']).name)}：{row['status']}，{row['elapsed_seconds']:.2f} 秒，{row['code_cells']} 个代码单元，{row['outputs']} 项输出。"
        for row in execution
    ) or "讲师 notebook 执行尚未完成。"
    test_log=(OUT/"tests.log").read_text(encoding="utf-8") if (OUT/"tests.log").exists() else ""
    test_result=next((line.strip() for line in reversed(test_log.splitlines()) if " passed" in line),"测试状态尚未确认")
    classroom_section=(f"\n## 课堂数据补充：拟合购买结果，是否学会了效应排序？\n\n{image('11_generalization_and_uplift')}\n\n左图比较训练和独立留出集上的 outcome AUC，右图使用随机 pilot 检查效应排序。较大的训练/测试落差支持过拟合诊断；AUC 与 Qini 仍是不同目标。这里保留原课堂 7k/3k 划分，与上面的独立模拟矩阵分开报告。\n" if (FIG/'11_generalization_and_uplift.png').exists() else "")
    text=f"""# 因果 Café 实验审阅：更多数据、不同算法，什么时候能改善决策？

这是实际运行后的讲师审阅版。数据全部来自模拟器；普通 participant notebook **没有执行**。先读图，再决定 notebook 的教学内容。训练与模型加载的基础步骤只保留在代码和测试记录里。

## 先看本次实验支持什么

已完成 **{manifest['completed_runs']} 次候选模型实验**：主比较 180 次（3 个场景 × 3 个规模 × 5 个种子 × 4 种方法），隐藏变量恢复的配对诊断另有 60 次。一次实验包含训练、保存、加载与评估；T-Learner 内部训练两个基模型，Causal Forest 内部含多阶段拟合。完整批量实验墙钟时间为 **{manifest['elapsed_seconds']:.1f} 秒**，包含拟合、加载、预测和写出文件。

{chr(10).join(winners)}
- 隐藏混杂场景中，50k 行的 T-logistic 平均 ATE 偏差在 U 缺失时为 **{100*hidden:+.2f} 个百分点**，把同一批数据中的 U 提供给模型后为 **{100*observed:+.2f} 个百分点**。这是配对模拟诊断，不是现实数据的因果识别证明。

所有候选、种子和场景均按预先固定配置报告，没有从评估结果挑选种子或调参。图中阴影是 **5 个训练种子的最小至最大值**，不是置信区间，也没有覆盖全部评估抽样不确定性。

## 1. 三类方法究竟把问题拆在哪里？

S/T 是组织监督学习估计处理效应的方式，底层模型仍会限制它能表达的形状。本实验 S-Learner 使用显式 X×T 交互的 logistic；Causal Forest 使用 EconML 的 CausalForestDML，包含交叉拟合的结果/处理分配模型、残差化与 honest forest（[官方接口说明](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html)）。它的模型复杂度、正则化与底层学习器并未与其他方法完全等同，所以这里比较的是四个固定配置。

{image('01_effect_estimators')}

## 2. 样本更多，效应估计就一定更好吗？

纵轴是独立 5k 顾客上的 CATE RMSE，越低越好。三个场景下的真值只用于评估，从不进入拟合或超参数选择。每个种子先生成同一个 50k 训练池，再取嵌套的 2k/10k/50k 子集；不同规模共享评估集，因此曲线反映训练样本增加的变化。

{image('02_learning_curves')}

观察的是误差是否随数据增加下降、是否出现平台，以及方法之间的差距是否跨种子保持。单一模拟器中的数据增加不能验证业务机制是否真实。

## 3. 误差降低，是否转成同预算下的业务价值？

每条策略按预测增量利润排序，只选择预测正利润顾客，并遵守固定 2,500 预算。评价使用同一批 5k 目标顾客的模拟真值计算**期望**增量利润，不是随机生成一次购买后得到的实收收入。横线为按真值利润排序的贪心参考，不声明它是严格的背包全局最优解。

{image('03_policy_learning_curves')}

RMSE 与策略收益评价不同目标：整体误差小，并不保证预算边界附近顾客的排序最好。

## 4. 隐藏混杂：数据越多，也可能更稳定地偏离

{image('10_hidden_confounding_dag')}

这个场景里的 U 同时影响发券和基础购买概率，但处理效应的概率差只依赖可见 X。因此下面的误差差异不会被解释成“缺失了一个额外的效应修饰变量”。恢复 U 的配对实验保持顾客、标签、种子、预算与模型配置相同，仅改变模型可用特征。

{image('04_hidden_variable_audit')}

连线连接同一种方法、同一种子的两种可用信息条件。恢复 U 后仍可能有模型近似误差和有限样本误差，不能要求所有点都精确回到零。

## 5. 随机 pilot 上，哪种排序把更受影响的人排在前面？

{image('05_randomized_pilot_qini')}

这里固定显示 10k 训练行、seed 42，方便检查单次完整曲线；它不是挑出的最佳种子，学习曲线包含全部五个种子。各场景内部共用同一个独立 5k 随机 pilot。Qini 衡量排序，不直接衡量利润；更换 pilot 样本量后不能直接比较这里的原始 Qini 数值。

## 6. 非线性交互，模型到底学到了什么形状？

{image('06_effect_surface')}

每个网格是同一批目标顾客中、price sensitivity 与 offer affinity 分箱后的平均效应。虚线标出模拟机制中的两个阈值；其余变量在箱内平均。这是预测与真值的形状比较，不是把图上单个变量解释为可干预的因果原因。空白格表示没有顾客。

## 7. 预测 uplift 的绝对值是否可信？

{image('07_effect_calibration')}

每种方法按自身预测分成十组，再比较组内预测与真值均值。对角线代表组均值校准；偏离说明估计效应的绝对值与真值不一致。分组均值可能掩盖个体层面的误差，因此与 RMSE 一起阅读。

## 8. 预算改变，策略选择会不会改变？

{image('08_budget_sensitivity')}

使用 nonlinear 场景、10k 训练行、seed 42 的已保存预测，预算从 0 到 4,000 改变；没有重新训练模型。这张图检查固定排名在不同决策约束下的表现，不能用来事后挑一个最有利预算再宣称胜出。

## 9. 增加复杂度，带来什么成本与取舍？

{image('09_cost_and_decision_tradeoff')}

左图为 CPU 拟合墙钟时间：线是跨场景与种子的中位数，阴影为最小至最大值。排除了首次导入、保存和预测。部分初始实验与测试共享过 CPU，且机器有其他系统工作，时间属于本机实测，不能当作隔离硬件基准。右图的 Qini 与利润来自独立 pilot 和固定目标人群，相关趋势不意味着两指标等价。

{classroom_section}

## 10. 一万行训练时的精确对照

以下是五个种子的均值（耗时列为中位数）；其余规模、每个种子和恢复 U 的全部记录见 {link('metrics.csv','完整 metrics.csv')}。

{markdown_table(pd.DataFrame(rows))}

## 可复查的证据与限制

- 测试：**{test_result}**，见 {link('tests.log','测试原始输出')}。覆盖两份答案流程中 logistic/boosting 的重复拟合防护、加载对象用于后续预测、特征顺序、S-Learner 和真实 Causal Forest 的保存/加载、数据独立性、预算、全部 240 组指标独立重算，以及 notebook 源码/真实执行输出。
- Qini 重复训练已移除；后续评估和主策略使用加载后的模型预测。源码改动与实验验证分开留档，训练/加载的基础操作不再单独画教学流程图。
- 配置与源文件哈希：{link('manifest.json','manifest.json')}；依赖版本：{link('environment.lock.txt','environment.lock.txt')}。
- 每个实验都有真实拟合的模型、版本/特征元数据、目标顾客预测、pilot 预测和独立指标 JSON，保存在 `runs/`。加载前后预测误差的最大值为 {metrics.reload_max_abs_difference.max():.3g}（容差 1e-12）。
- S/T/Causal Forest 均在 CPU 上运行；forest 为 120 棵树、3-fold cross-fitting、honest=True、最多 2 个并行工作线程。本实验 inference=False，**没有生成 Causal Forest 个体效应置信区间**。
- 样本量指实际参与拟合的训练行数；独立目标人群和随机 pilot 分别固定为 5,000 行，预算始终 2,500。每个场景的评估与 pilot 指纹跨方法/种子/规模一致，训练与评估 ID 命名空间不相交。
- participant 主数据未扩容；模拟 truth 和 latent U 只出现在讲师实验文件。结果支持这些指定模拟场景中的比较，不能当作真实业务增量效果证明。
- 本地修改尚未 commit/push，公开 main/Colab 链接未更新。普通 workshop 保留四个 TODO，并已同步源码；执行证据另存于 `executed/`。

### 讲师 notebook 的实际执行结果

{execution_lines}

### 建议你检查 notebook 时关注的内容

- 主线保留一万行和 T-Learner 动手实现；S-Learner 提供共享/分开结果模型的对照。这里的交互 logistic S 与 logistic T 有相近的表达能力，所以二者结果接近本身也是可解释的观察。
- Causal Forest 及样本规模、种子与场景比较放在讲师附录；按误差、校准、排名、收益四个问题展开，避免把一次排名说成方法的普遍优劣。
- 隐藏 U 与恢复 U 的配对图应保留，它把统计学习能力与识别条件分开检验。
- participant 在 Colab 上的真实浏览器下载仍为 **NOT RUN**；下载单元调用与 ZIP 内容有隔离测试。发布时需要一起带上新增的 `src/causal_models.py`。

## 方法依据

- [Künzel 等：S/T/X meta-learners](https://arxiv.org/html/1706.03461v4)：使用不同组织方式估计条件平均处理效应，没有一种方法在所有数据机制下都最好。
- [EconML CausalForestDML 官方文档](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html)：残差化、cross-fitting、honesty 和推断参数。
- [Wager 与 Athey：Causal Forest](https://arxiv.org/abs/1510.04342)：因果效应异质性的森林估计与相应识别、推断条件。
"""
    (OUT/"REVIEW.md").write_text(text,encoding="utf-8")
    agg.to_csv(OUT/"summary_metrics.csv",index=False)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete":
        raise RuntimeError("Refusing to present an unfinished study as complete")
    metrics=pd.read_csv(OUT/"metrics.csv")
    if len(metrics)!=240 or metrics.duplicated(["scenario","training_rows","seed","method","reveal_u"]).any():
        raise ValueError("The experiment grid is missing cases or has duplicates")
    main_grid=metrics[~metrics.reveal_u]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlepad":12})
    mechanisms()
    learning_curves(main_grid,"cate_rmse","Held-out CATE RMSE (lower is better)","02_learning_curves")
    learning_curves(main_grid,"true_policy_profit","True expected incremental profit ($)","03_policy_learning_curves")
    hidden_audit(metrics); qini_panels(); effect_maps(); calibration(); budget_curves()
    runtime_and_tradeoff(main_grid); hidden_dag(); classroom_generalization(); report(metrics,manifest)
    print("Rendered figure sets (PNG + SVG) and",OUT/"REVIEW.md")


if __name__=="__main__":
    main()
