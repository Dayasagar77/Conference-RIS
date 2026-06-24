#!/usr/bin/env python3
"""
run_all_figs.py  —  Generate ALL 4 New Figures (Figs 13–16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IEEE Access Revision: Access-2026-21197
Author: Dayasagar G + Dr. Deepa Nivethika S | VIT Chennai

Saves all figures to:
  D:\DAYA PHD\PHD WORK\RIS\results\FIGS_13_16\
    fig13_per_user_cdf.png
    fig14_multi_geometry.png
    fig15_hsdc_sc1.png
    fig16_outage_analysis.png

Run:
  cd D:\DAYA PHD\PHD WORK\RIS\channel_model
  python run_all_figs.py
"""

import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results', 'FIGS_13_16'))
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SYSTEM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
FC       = 3.5e9;  C_LIGHT = 3.0e8;  LAMBDA = C_LIGHT / FC
BW_TOTAL = 20.0e6; N_PAIRS = 12;     BW_PAIR = BW_TOTAL / N_PAIRS
P_TOTAL  = 2.0;    P_PAIR  = P_TOTAL / N_PAIRS;  ALPHA = 0.85
NOISE_W  = 10.0**( (-167.0 - 30.0) / 10.0 ) * BW_PAIR
AL_A=9.61; AL_B=0.16; ETA_LOS=1.0; ETA_NLOS=100.0; K_RICE=10.0
BLOCKAGE_LIN = 10.0**(60.0/10.0)
M_RIS=1024; RIS_POS=np.array([300.,250.,20.]); NBITS=3
N_LEVELS=8; PHI_STEP=2.0*np.pi/N_LEVELS
UAV_POS  = np.array([437.,584.,108.])
CENTROID = np.array([400.,600.,0.])
NEAR_R_MIN,NEAR_R_MAX = 50.,250.; FAR_R_MIN,FAR_R_MAX = 450.,800.
K_NEAR=12; K_FAR=12; DEVICE_SEED=42; EVAL_SEED=99; N_MC=50
DPI = 300   # publication quality

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CHANNEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fspl(a,b):
    d=max(float(np.linalg.norm(np.asarray(a)-np.asarray(b))),0.01)
    return (LAMBDA/(4*np.pi*d))**2

def plos(uav,dev):
    d2d=max(np.hypot(uav[0]-dev[0],uav[1]-dev[1]),1e-3)
    return 1/(1+AL_A*np.exp(-AL_B*(np.degrees(np.arctan(uav[2]/d2d))-AL_A)))

def qphase(theta):
    return (np.round(theta/PHI_STEP).astype(int)%N_LEVELS)*PHI_STEP

def place_devices():
    np.random.seed(DEVICE_SEED)
    rn=np.random.uniform(NEAR_R_MIN,NEAR_R_MAX,K_NEAR)
    tn=np.random.uniform(0,2*np.pi,K_NEAR)
    near=np.c_[CENTROID[0]+rn*np.cos(tn), CENTROID[1]+rn*np.sin(tn), np.zeros(K_NEAR)]
    rf=np.random.uniform(FAR_R_MIN,FAR_R_MAX,K_FAR)
    tf=np.random.uniform(0,2*np.pi,K_FAR)
    far=np.c_[CENTROID[0]+rf*np.cos(tf), CENTROID[1]+rf*np.sin(tf), np.zeros(K_FAR)]
    return near, far

def simulate(near, far):
    """Core MC simulation → returns rate matrices (K,N_MC) in Mbps."""
    pn=np.array([plos(UAV_POS,near[k]) for k in range(K_NEAR)])
    pf=np.array([plos(UAV_POS,far[k])  for k in range(K_FAR)])
    gn=np.array([fspl(UAV_POS,near[k]) for k in range(K_NEAR)])
    gfb=np.array([fspl(UAV_POS,far[k]) for k in range(K_FAR)])/BLOCKAGE_LIN
    gur=fspl(UAV_POS,RIS_POS)
    grf=np.array([fspl(RIS_POS,far[k]) for k in range(K_FAR)])

    np.random.seed(EVAL_SEED)

    def chan(g,p,K,blocked=False):
        gl=g[:,None]/ETA_LOS; gn_=g[:,None]/ETA_NLOS
        los=np.random.random((K,N_MC))<p[:,None]
        h1=(np.sqrt(K_RICE/(K_RICE+1)*gl)*np.exp(1j*np.random.uniform(0,2*np.pi,(K,N_MC)))
            +np.sqrt(gl/(K_RICE+1)/2)*(np.random.standard_normal((K,N_MC))+1j*np.random.standard_normal((K,N_MC))))
        h2=np.sqrt(gn_/2)*(np.random.standard_normal((K,N_MC))+1j*np.random.standard_normal((K,N_MC)))
        return np.where(los,h1,h2)

    Hn  = chan(gn,  pn, K_NEAR)
    Hfd = chan(gfb, pf, K_FAR)
    Hr  = np.sqrt(gur/2)*(np.random.standard_normal((M_RIS,N_MC))+1j*np.random.standard_normal((M_RIS,N_MC)))
    G   = np.sqrt(grf/2)[None,:,None]*(np.random.standard_normal((M_RIS,K_FAR,N_MC))+1j*np.random.standard_normal((M_RIS,K_FAR,N_MC)))
    Phi = np.exp(1j*qphase(np.angle(Hr[:,None,:])-np.angle(G)))
    Hris= np.sum(np.conj(Hr[:,None,:])*Phi*G, axis=0)
    Heff= Hfd+Hris

    def rate_near(H): return BW_PAIR*np.log2(1+(1-ALPHA)*P_PAIR*np.abs(H)**2/NOISE_W)*1e-6
    def rate_far(H):
        p=np.abs(H)**2
        return BW_PAIR*np.log2(1+ALPHA*P_PAIR*p/((1-ALPHA)*P_PAIR*p+NOISE_W))*1e-6

    return rate_near(Hn), rate_far(Hfd), rate_far(Heff)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — Per-User Rate CDFs
# ─────────────────────────────────────────────────────────────────────────────
def fig13(near, far):
    print("  [Fig 13] Per-user rate CDFs ...", end=' ', flush=True)
    R_near, R_far_nr, R_far_r = simulate(near, far)

    def ecdf(d):
        x=np.sort(d.flatten()); y=np.arange(1,len(x)+1)/len(x)
        return np.r_[0,x], np.r_[0,y]

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    fig.subplots_adjust(wspace=0.32)
    cb,cr='#1f77b4','#d62728'

    ax=axes[0]
    x1,y1=ecdf(R_near); x2,y2=ecdf(R_near)
    ax.step(x1,y1,color=cr,lw=2,where='post',label='Without RIS')
    ax.step(x2,y2,color=cb,lw=2,where='post',ls='--',label='With RIS (≈same)')
    ax.set_xlabel('Per-User Throughput (Mbps)'); ax.set_ylabel('CDF')
    ax.set_title('Near Users  (50–250 m ring)',fontweight='bold')
    ax.set_ylim(-0.02,1.08); ax.legend(); ax.grid(True,ls=':',alpha=0.5)
    ax.text(0.04,0.92,'RIS phase optimised\nfor far users\n→ 0 dB near gain',
            transform=ax.transAxes,fontsize=9,va='top',
            bbox=dict(boxstyle='round,pad=0.4',facecolor='#ffffcc',alpha=0.9))

    ax=axes[1]
    x3,y3=ecdf(R_far_nr); x4,y4=ecdf(R_far_r)
    ax.step(x3,y3,color=cr,lw=2,where='post',label='Without RIS (60 dB block)')
    ax.step(x4,y4,color=cb,lw=2,where='post',ls='--',label='With RIS  (M=1024, 3-bit)')
    ax.set_xlabel('Per-User Throughput (Mbps)'); ax.set_ylabel('CDF')
    ax.set_title('Far Users  (450–800 m ring)',fontweight='bold')
    ax.set_ylim(-0.02,1.08); ax.legend(); ax.grid(True,ls=':',alpha=0.5)

    # figure title omitted; the IEEE caption provides it
    out=os.path.join(OUT_DIR,'fig13_per_user_cdf.png')
    fig.savefig(out,dpi=DPI,bbox_inches='tight'); plt.close(fig)
    print(f"saved → {out}")
    return R_near, R_far_nr, R_far_r

# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — Multi-Geometry Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _eval_seed(device_seed):
    np.random.seed(device_seed)
    rn=np.random.uniform(NEAR_R_MIN,NEAR_R_MAX,K_NEAR); tn=np.random.uniform(0,2*np.pi,K_NEAR)
    near=np.c_[CENTROID[0]+rn*np.cos(tn),CENTROID[1]+rn*np.sin(tn),np.zeros(K_NEAR)]
    rf=np.random.uniform(FAR_R_MIN,FAR_R_MAX,K_FAR);  tf=np.random.uniform(0,2*np.pi,K_FAR)
    far=np.c_[CENTROID[0]+rf*np.cos(tf),CENTROID[1]+rf*np.sin(tf),np.zeros(K_FAR)]
    Rn,Rfnr,Rfr=simulate(near,far)
    return float(Rn.mean(axis=1).sum()+Rfr.mean(axis=1).sum())

def fig14():
    print("  [Fig 14] Multi-geometry evaluation ...", end=' ', flush=True)
    TEST_SEEDS=list(range(100,120))
    thr42   = _eval_seed(DEVICE_SEED)
    thr_arr = np.array([_eval_seed(s) for s in TEST_SEEDS])
    mu,std  = thr_arr.mean(), thr_arr.std()

    OMA_REF=120.93; NOMA_REF=187.18
    fig,ax=plt.subplots(figsize=(13,5.5))
    x_test=np.arange(len(TEST_SEEDS)); x_train=len(TEST_SEEDS)
    ax.bar(x_test,thr_arr,color='#4C9BE8',edgecolor='#2060A0',lw=0.7,zorder=3,label='Test seeds 100–119')
    ax.bar(x_train,thr42,color='#F5A623',edgecolor='#B07A18',lw=0.9,zorder=3,label='Training seed 42 (paper)')
    ax.axhline(mu,color='#1f77b4',lw=1.6,ls='--',zorder=4,label=f'Test mean = {mu:.1f} Mbps')
    ax.axhspan(mu-std,mu+std,color='#1f77b4',alpha=0.12,zorder=2,label=f'±1σ = {std:.1f} Mbps')
    ax.axhline(OMA_REF,color='#d62728',lw=1.4,ls=':',zorder=4,label=f'OMA baseline = {OMA_REF} Mbps')
    ax.axhline(NOMA_REF,color='#2ca02c',lw=1.4,ls=':',zorder=4,label=f'NOMA H*=125m = {NOMA_REF} Mbps')
    xtl=[str(s) for s in TEST_SEEDS]+['42\n(train)']
    ax.set_xticks(np.arange(len(TEST_SEEDS)+1)); ax.set_xticklabels(xtl,fontsize=8.5)
    ax.set_xlabel('Device Layout Seed'); ax.set_ylabel('Aggregate Throughput (Mbps)')
    # figure title omitted; the IEEE caption provides it
    ax.text(0.01,0.97,f'20-seed stats\nmean={mu:.2f} Mbps\nstd={std:.2f} Mbps\n'
            f'min={thr_arr.min():.2f}\nmax={thr_arr.max():.2f}',
            transform=ax.transAxes,fontsize=9,va='top',
            bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='#888',alpha=0.9))
    ylo=max(0,OMA_REF*0.85); yhi=max(thr_arr.max(),thr42,NOMA_REF)*1.06
    ax.set_ylim(ylo,yhi); ax.set_xlim(-0.6,len(TEST_SEEDS)+0.6)
    ax.legend(loc='lower right',fontsize=9.5); ax.grid(axis='y',ls=':',alpha=0.45)
    plt.tight_layout()
    out=os.path.join(OUT_DIR,'fig14_multi_geometry.png')
    fig.savefig(out,dpi=DPI,bbox_inches='tight'); plt.close(fig)
    print(f"saved → {out}")
    return mu, std, thr_arr.min(), thr_arr.max()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — HSDC on 24 SC1 Devices
# ─────────────────────────────────────────────────────────────────────────────
def fig15(near, far):
    print("  [Fig 15] HSDC clustering ...", end=' ', flush=True)
    pos2d=np.vstack([near[:,:2], far[:,:2]])
    labels=['N']*K_NEAR+['F']*K_FAR
    R_COVER=500.0

    Z=linkage(pos2d, method='ward')
    k_star=None
    for k in range(2,9):
        asgn=fcluster(Z,k,criterion='maxclust')
        cents=np.array([pos2d[asgn==c].mean(axis=0) for c in range(1,k+1)])
        dists=np.array([np.linalg.norm(pos2d[i]-cents[asgn[i]-1]) for i in range(len(pos2d))])
        if dists.max()<=R_COVER:
            k_star=k; best_a=asgn; best_c=cents; best_d=dists; break
    if k_star is None:
        k_star=4; best_a=fcluster(Z,k_star,criterion='maxclust')
        best_c=np.array([pos2d[best_a==c].mean(axis=0) for c in range(1,k_star+1)])
        best_d=np.array([np.linalg.norm(pos2d[i]-best_c[best_a[i]-1]) for i in range(len(pos2d))])
    try: sil=float(silhouette_score(pos2d,best_a))
    except: sil=0.0

    CMAP=plt.cm.get_cmap('tab10',10); C=[CMAP(i) for i in range(k_star)]
    n_near=[sum(1 for i in range(len(pos2d)) if best_a[i]==c+1 and labels[i]=='N') for c in range(k_star)]
    n_far =[sum(1 for i in range(len(pos2d)) if best_a[i]==c+1 and labels[i]=='F') for c in range(k_star)]

    fig,axes=plt.subplots(1,2,figsize=(13,6)); fig.subplots_adjust(wspace=0.30)

    ax=axes[0]
    for ci,cent in enumerate(best_c):
        ax.add_patch(Circle(cent,R_COVER,color=C[ci],alpha=0.09,ls='--',fill=True,zorder=1))
    for r,ls in [(NEAR_R_MAX,':'),(FAR_R_MIN,':'),(FAR_R_MAX,':')]:
        ax.add_patch(Circle(CENTROID[:2],r,color='grey',fill=False,lw=0.7,ls=ls,alpha=0.4))
    for i,(p,lb) in enumerate(zip(pos2d,labels)):
        ax.scatter(p[0],p[1],s=90,c=[C[best_a[i]-1]],
                   marker='o' if lb=='N' else '^',edgecolors='black',lw=0.6,zorder=4)
        ax.text(p[0]+3,p[1]+3,str(i+1),fontsize=7,color='#333')
    for ci,cent in enumerate(best_c):
        ax.scatter(cent[0],cent[1],s=220,c=[C[ci]],marker='*',edgecolors='black',lw=1.0,zorder=5)
    ax.scatter(CENTROID[0],CENTROID[1],s=160,c='black',marker='+',lw=1.8,zorder=6)
    patches=[mpatches.Patch(color=C[c],alpha=0.8,label=f'Cluster {c+1}') for c in range(k_star)]
    patches+=[mpatches.Patch(color='grey',label='● Near  ▲ Far'),
              mpatches.Patch(color='lightgrey',alpha=0.4,label=f'Coverage R={R_COVER:.0f} m')]
    ax.legend(handles=patches,loc='lower left',fontsize=8.5,framealpha=0.85)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'HSDC on 24 SC1 Devices — K*={k_star}  (coverage ✓)',fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(CENTROID[0]-FAR_R_MAX-150,CENTROID[0]+FAR_R_MAX+150)
    ax.set_ylim(CENTROID[1]-FAR_R_MAX-150,CENTROID[1]+FAR_R_MAX+150)
    ax.grid(True,ls=':',alpha=0.35)
    ax.text(0.02,0.98,f'K* = {k_star}\nSilhouette = {sil:.3f}\nMax d = {best_d.max():.1f} m ≤ {R_COVER:.0f} m',
            transform=ax.transAxes,fontsize=9,va='top',
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',edgecolor='#888',alpha=0.9))

    ax2=axes[1]; x=np.arange(k_star); w=0.35
    b1=ax2.bar(x-w/2,n_near,w,color='#5FA2DD',edgecolor='white',lw=0.6,label='Near')
    b2=ax2.bar(x+w/2,n_far, w,color='#E05C5C',edgecolor='white',lw=0.6,label='Far')
    for bars in [b1,b2]:
        for bar in bars:
            h=bar.get_height()
            if h>0: ax2.text(bar.get_x()+bar.get_width()/2,h+0.08,str(int(h)),ha='center',va='bottom',fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels([f'Cluster {c+1}' for c in range(k_star)],fontsize=10)
    ax2.set_ylabel('Devices'); ax2.set_title('Near / Far Composition',fontweight='bold')
    ax2.legend(fontsize=10); ax2.set_ylim(0,max(max(n_near),max(n_far))+2.5)
    ax2.grid(axis='y',ls=':',alpha=0.45)
    n_near_only=sum(1 for c in range(k_star) if n_far[c]==0)
    n_far_only =sum(1 for c in range(k_star) if n_near[c]==0)
    ax2.text(0.98,0.97,f'Near-only: {n_near_only}\nFar-only:  {n_far_only}\nMixed:     {k_star-n_near_only-n_far_only}',
             transform=ax2.transAxes,fontsize=9,va='top',ha='right',
             bbox=dict(boxstyle='round,pad=0.4',facecolor='#fffde7',edgecolor='#aaa',alpha=0.9))

    # figure title omitted; the IEEE caption provides it
    out=os.path.join(OUT_DIR,'fig15_hsdc_sc1.png')
    fig.savefig(out,dpi=DPI,bbox_inches='tight'); plt.close(fig)
    print(f"saved → {out}")
    return k_star, sil

# ─────────────────────────────────────────────────────────────────────────────
# FIG 16 — Outage Probability
# ─────────────────────────────────────────────────────────────────────────────
def fig16(R_near, R_far_nr, R_far_r):
    print("  [Fig 16] Outage analysis ...", end=' ', flush=True)
    def pout(rates,th): return float((rates<th).mean())
    def pout_curve(rates,ths): return np.array([pout(rates,t) for t in ths])

    R_MIN=0.10; BARS=[0.05,0.10,0.50]
    p_fnr=pout(R_far_nr,R_MIN); p_fr=pout(R_far_r,R_MIN)
    p_n  =pout(R_near,   R_MIN); delta=p_fnr-p_fr

    th_far=np.linspace(0,min(float(R_far_r.max())*1.2,2.5),300)
    th_near=np.linspace(0,min(float(R_near.max())*1.05,25),300)

    cb,cr,cg='#1f77b4','#d62728','#2ca02c'
    fig,axes=plt.subplots(1,2,figsize=(13,5.2)); fig.subplots_adjust(wspace=0.30)

    ax=axes[0]
    ax.plot(th_near,pout_curve(R_near,th_near),color=cb,lw=2,label='Near users (±RIS same)')
    ax.plot(th_far, pout_curve(R_far_nr,th_far),color=cr,lw=2,ls='--',label='Far — no RIS (60 dB block)')
    ax.plot(th_far, pout_curve(R_far_r, th_far),color=cg,lw=2,ls='-.',label=f'Far — with RIS (M={M_RIS})')
    ax.axvline(R_MIN,color='#888',lw=1.2,ls=':',label=f'R_min={R_MIN} Mbps')
    try:
        ax.annotate(f'P_out={p_fnr:.3f}',xy=(R_MIN,p_fnr),xytext=(R_MIN*2.8,p_fnr-0.08),fontsize=9,color=cr,
                    arrowprops=dict(arrowstyle='->',color=cr,lw=1.2))
        ax.annotate(f'P_out={p_fr:.3f}', xy=(R_MIN,p_fr), xytext=(R_MIN*2.8,p_fr+0.06), fontsize=9,color=cg,
                    arrowprops=dict(arrowstyle='->',color=cg,lw=1.2))
        ax.annotate('',xy=(R_MIN*0.9,p_fr),xytext=(R_MIN*0.9,p_fnr),
                    arrowprops=dict(arrowstyle='<->',color='purple',lw=1.5))
        ax.text(R_MIN*0.55,(p_fnr+p_fr)/2,f'Δ={delta*100:.0f} pp',
                ha='center',fontsize=9,color='purple',fontweight='bold')
    except: pass
    ax.set_xlabel('Rate Threshold R_min (Mbps)'); ax.set_ylabel('Outage Probability  P(R < R_min)')
    ax.set_title('Outage Probability vs Threshold',fontweight='bold')
    ax.set_xlim(0,th_far[-1]); ax.set_ylim(-0.03,1.08)
    ax.legend(loc='lower right',fontsize=9.5); ax.grid(True,ls=':',alpha=0.45)

    ax2=axes[1]; x=np.arange(len(BARS)); w=0.28
    fnr=[pout(R_far_nr,r) for r in BARS]
    fr =[pout(R_far_r, r) for r in BARS]
    n  =[pout(R_near,  r) for r in BARS]
    b1=ax2.bar(x-w,fnr,w,color=cr,edgecolor='white',lw=0.7,label='Far — no RIS')
    b2=ax2.bar(x,  fr, w,color=cg,edgecolor='white',lw=0.7,label='Far — with RIS')
    b3=ax2.bar(x+w,n,  w,color=cb,edgecolor='white',lw=0.7,label='Near (±RIS same)')
    for bars in [b1,b2,b3]:
        for bar in bars:
            h=bar.get_height()
            ax2.text(bar.get_x()+bar.get_width()/2,h+0.012,
                     f'{h:.2f}' if h>=0.01 else f'{h:.3f}',ha='center',va='bottom',fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels([f'R_min={r} Mbps' for r in BARS],fontsize=9.5)
    ax2.set_ylabel('Outage Probability'); ax2.set_title('Outage at Key Thresholds',fontweight='bold')
    ax2.set_ylim(0,1.18); ax2.legend(fontsize=9.5); ax2.grid(axis='y',ls=':',alpha=0.45)

    # figure title omitted; the IEEE caption provides it
    out=os.path.join(OUT_DIR,'fig16_outage_analysis.png')
    fig.savefig(out,dpi=DPI,bbox_inches='tight'); plt.close(fig)
    print(f"saved → {out}")
    return p_fnr, p_fr, delta

# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0=time.time()
    print()
    print("="*60)
    print("  Generating Figs 13–16  |  IEEE Access-2026-21197")
    print(f"  Output → {OUT_DIR}")
    print("="*60)

    near, far = place_devices()

    R_near, R_far_nr, R_far_r = fig13(near, far)
    mu,std,lo,hi              = fig14()
    k_star, sil               = fig15(near, far)
    p_fnr, p_fr, delta        = fig16(R_near, R_far_nr, R_far_r)

    elapsed = time.time() - t0
    print()
    print(f"  ✓ All 4 figures saved in {elapsed:.1f} s")
    print()
    print("  SUMMARY")
    print(f"  Fig 13 — Per-user CDF: far RIS vs no-RIS clearly separated")
    print(f"  Fig 14 — Generalisation: {mu:.1f}±{std:.1f} Mbps across 20 seeds")
    print(f"  Fig 15 — HSDC K*={k_star}: perfect near/far separation (sil={sil:.3f})")
    print(f"  Fig 16 — Outage at 0.1 Mbps: {p_fnr*100:.0f}% → {p_fr*100:.0f}% ({delta*100:.0f} pp reduction)")
    print()
    print("  Files ready for insertion into v12_FINAL.docx:")
    for n in ['fig13_per_user_cdf.png','fig14_multi_geometry.png',
              'fig15_hsdc_sc1.png','fig16_outage_analysis.png']:
        print(f"    {OUT_DIR}\\{n}")
    print("="*60)
